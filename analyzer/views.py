import os
import json
import pdfplumber
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def index(request):
    return render(request, 'analyzer/index.html')


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def analyze_with_ai(resume_text, job_description):
    prompt = f"""You are an expert resume analyst and career coach.

Analyze this resume against the job description and return ONLY a JSON object with no markdown, no backticks, no extra text.

Resume:
{resume_text}

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
    prompt = f"""You are an expert career coach analyzing a developer's resume.

Resume:
{resume_text}

Analyze this resume and return ONLY a JSON object with no markdown, no backticks, no extra text.

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

Evaluate these specific roles in this order:
1. Junior Django Developer
2. Junior Python Developer
3. Junior Backend Developer
4. Junior Full Stack Developer
5. Junior Software Engineer
6. Python/Django Freelancer
7. Junior AI/ML Engineer
8. Junior Data Analyst
9. Junior DevOps Engineer
10. Junior React Developer

Be honest and accurate. Base scores purely on what is in the resume."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace('```json', '').replace('```', '').strip()
    return json.loads(raw)


@csrf_exempt
def analyze(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    job_description = request.POST.get('job_description', '').strip()
    resume_file = request.FILES.get('resume')

    if not job_description or not resume_file:
        return JsonResponse({'error': 'Both resume and job description are required'}, status=400)

    try:
        resume_text = extract_text_from_pdf(resume_file)
        result = analyze_with_ai(resume_text, job_description)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def career_fit(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    resume_file = request.FILES.get('resume')

    if not resume_file:
        return JsonResponse({'error': 'Resume is required'}, status=400)

    try:
        resume_text = extract_text_from_pdf(resume_file)
        result = career_fit_with_ai(resume_text)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)