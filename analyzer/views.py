import os
import json
import stripe
import pdfplumber
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
stripe.api_key = settings.STRIPE_SECRET_KEY

FREE_ANALYSES = 3
CREDITS_PER_PACK = 10


def index(request):
    # Initialize session credits for new visitors
    if 'analyses_remaining' not in request.session:
        request.session['analyses_remaining'] = FREE_ANALYSES
    return render(request, 'analyzer/index.html', {
        'stripe_publishable_key': settings.STRIPE_PUBLISHABLE_KEY
    })


def get_credits(request):
    remaining = request.session.get('analyses_remaining', FREE_ANALYSES)
    return JsonResponse({'credits': remaining})


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


def analyze_with_ai(collection, job_description):
    job_embedding = get_embedding(job_description)
    results = collection.query(
        query_embeddings=[job_embedding],
        n_results=3
    )
    retrieved_chunks = results['documents'][0]
    context = " ".join(retrieved_chunks)

    prompt = f"""You are an expert resume analyst and career coach.

Analyze this resume against the job description and return ONLY a JSON object with no markdown, no backticks, no extra text.

Resume:
{context}

Job Description:
{job_description}

Return exactly this JSON structure:
{{
  "score": <integer 0-100>,
  "present_skills": [<list of skills from job description found in resume>],
  "missing_skills": [<list of skills from job description NOT found in resume>],
  "cover_letter": "<a professional 3 paragraph cover letter tailored to this specific job>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace('```json', '').replace('```', '').strip()
    return json.loads(raw)


def career_fit_with_ai(resume_text):
    prompt = f"""You are an expert career coach analyzing a resume.

Resume:
{resume_text}

Analyze this resume and return ONLY a JSON object with no markdown, no backticks, no extra text.

First, identify the candidate's field and experience level based on the resume content.
Then evaluate the 10 most relevant job roles for this specific person — do NOT assume they are in tech.
Consider any industry: healthcare, education, trades, business, creative, legal, finance, etc.

Return exactly this JSON structure:
{{
  "roles": [
    {{
      "title": "<job title>",
      "match": <integer 0-100>,
      "notes": "<one sentence explaining why this is or isn't a good fit>"
    }}
  ]
}}

Be honest and accurate. Base scores purely on what is in the resume. Return exactly 10 roles."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace('```json', '').replace('```', '').strip()
    return json.loads(raw)


def use_credit(request):
    """Deduct one credit. Returns True if allowed, False if out of credits."""
    remaining = request.session.get('analyses_remaining', FREE_ANALYSES)
    if remaining <= 0:
        return False
    request.session['analyses_remaining'] = remaining - 1
    request.session.modified = True
    return True


@csrf_exempt
def create_checkout_session(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': settings.STRIPE_PRICE_ID,
                'quantity': 1,
            }],
            mode='payment',
            success_url=request.build_absolute_uri('/payment-success/') + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=request.build_absolute_uri('/payment-cancel/'),
            metadata={'django_session_key': request.session.session_key}
        )
        return JsonResponse({'url': checkout_session.url})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def payment_success(request):
    # Stripe redirects here after payment — credits added via webhook
    return render(request, 'analyzer/payment_success.html')


def payment_cancel(request):
    return render(request, 'analyzer/payment_cancel.html')


@csrf_exempt
@require_POST
def stripe_webhook(request):
    payload = request.body
    sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        return JsonResponse({'error': 'Invalid payload'}, status=400)
    except stripe.error.SignatureVerificationError:
        return JsonResponse({'error': 'Invalid signature'}, status=400)

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        metadata = session.get('metadata') or {}
        django_session_key = metadata.get('django_session_key')

        print(f"Metadata: {metadata}")
        print(f"Session key from metadata: {django_session_key}")

        if django_session_key:
            try:
                from django.contrib.sessions.backends.db import SessionStore
                s = SessionStore(session_key=django_session_key)
                s.load()
                current = s.get('analyses_remaining', 0)
                print(f"Current credits before update: {current}")
                s['analyses_remaining'] = current + CREDITS_PER_PACK
                s.save()
                print(f"Credits updated to: {s['analyses_remaining']}")
            except Exception as e:
                print(f"Session update error: {e}")
                return JsonResponse({'error': str(e)}, status=500)
        else:
            print("No django_session_key found in metadata!")

    return JsonResponse({'status': 'ok'})


@csrf_exempt
def analyze(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    if not use_credit(request):
        return JsonResponse({'error': 'no_credits'}, status=402)

    job_description = request.POST.get('job_description', '').strip()
    resume_file = request.FILES.get('resume')

    if not job_description or not resume_file:
        return JsonResponse({'error': 'Both resume and job description are required'}, status=400)

    try:
        resume_text = extract_text_from_pdf(resume_file)
        chunks = resume_text.split(". ")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="resume_chunks")

        embeddings = [get_embedding(chunk) for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.add(documents=chunks, embeddings=embeddings, ids=ids)

        result = analyze_with_ai(collection, job_description)
        result['credits_remaining'] = request.session.get('analyses_remaining', 0)
        return JsonResponse(result)

    except Exception as e:
        # Refund the credit if something went wrong
        request.session['analyses_remaining'] = request.session.get('analyses_remaining', 0) + 1
        request.session.modified = True
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def career_fit(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    if not use_credit(request):
        return JsonResponse({'error': 'no_credits'}, status=402)

    resume_file = request.FILES.get('resume')

    if not resume_file:
        return JsonResponse({'error': 'Resume is required'}, status=400)

    try:
        resume_text = extract_text_from_pdf(resume_file)
        result = career_fit_with_ai(resume_text)
        result['credits_remaining'] = request.session.get('analyses_remaining', 0)
        return JsonResponse(result)

    except Exception as e:
        request.session['analyses_remaining'] = request.session.get('analyses_remaining', 0) + 1
        request.session.modified = True
        return JsonResponse({'error': str(e)}, status=500)