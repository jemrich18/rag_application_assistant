import os
import json
import re
import pdfplumber
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

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


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


def analyze_with_ai(collection, job_description):
    # Embed the job description and retrieve relevant resume chunks
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


@csrf_exempt
def analyze(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    job_description = request.POST.get('job_description', '').strip()
    resume_file = request.FILES.get('resume')

    if not job_description or not resume_file:
        return JsonResponse({'error': 'Both resume and job description are required'}, status=400)

    try:
        # Extract text from resume PDF
        resume_text = extract_text_from_pdf(resume_file)

        # Chunk the resume
        chunks = resume_text.split(". ")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        # Initialize ChromaDB and store chunks
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="resume_chunks")

        # Embed and store each chunk
        embeddings = []
        for chunk in chunks:
            vector = get_embedding(chunk)
            embeddings.append(vector)

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

        # Analyze using RAG retrieval
        result = analyze_with_ai(collection, job_description)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def strip_html(text):
    return re.sub('<[^<]+?>', ' ', text or '')


def fetch_remotive_jobs(keyword, limit=25):
    try:
        resp = requests.get(
            'https://remotive.com/api/remote-jobs',
            params={'search': keyword} if keyword else {},
            timeout=10,
        )
        resp.raise_for_status()
        jobs = resp.json().get('jobs', [])[:limit]
        return [{
            'title': j.get('title', ''),
            'company': j.get('company_name', ''),
            'location': j.get('candidate_required_location', 'Remote'),
            'url': j.get('url', ''),
            'description': strip_html(j.get('description', ''))[:800],
            'source': 'Remotive',
        } for j in jobs]
    except requests.RequestException:
        return []


def fetch_arbeitnow_jobs(keyword, location, limit=25):
    try:
        resp = requests.get('https://www.arbeitnow.com/api/job-board-api', timeout=10)
        resp.raise_for_status()
        jobs = resp.json().get('data', [])
        keyword_lower = (keyword or '').lower()
        location_lower = (location or '').lower()
        matches = []
        for j in jobs:
            title = j.get('title', '')
            tags = ' '.join(j.get('tags', []) or [])
            haystack = f"{title} {tags}".lower()
            if keyword_lower and keyword_lower not in haystack:
                continue
            if location_lower and location_lower not in (j.get('location', '') or '').lower() and 'remote' not in location_lower:
                if not j.get('remote'):
                    continue
            matches.append({
                'title': title,
                'company': j.get('company_name', ''),
                'location': j.get('location') or ('Remote' if j.get('remote') else ''),
                'url': j.get('url', ''),
                'description': strip_html(j.get('description', ''))[:800],
                'source': 'Arbeitnow',
            })
            if len(matches) >= limit:
                break
        return matches
    except requests.RequestException:
        return []


def score_jobs_against_resume(resume_text, jobs):
    if not jobs:
        return []

    listing_block = "\n\n".join(
        f"[{i}] Title: {job['title']}\nCompany: {job['company']}\nDescription: {job['description']}"
        for i, job in enumerate(jobs)
    )

    prompt = f"""You are an expert career coach. Score how well each job listing matches the candidate's resume.

Resume:
{resume_text[:4000]}

Job Listings:
{listing_block}

Return ONLY a JSON object with no markdown, no backticks, no extra text, in exactly this structure:
{{
  "matches": [
    {{"index": <listing index>, "score": <integer 0-100>, "reason": "<one sentence why this is or isn't a good fit>"}}
  ]
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace('```json', '').replace('```', '').strip()
    parsed = json.loads(raw)

    scored = []
    for match in parsed.get('matches', []):
        idx = match.get('index')
        if idx is None or idx < 0 or idx >= len(jobs):
            continue
        job = dict(jobs[idx])
        job['score'] = match.get('score', 0)
        job['reason'] = match.get('reason', '')
        job.pop('description', None)
        scored.append(job)

    scored.sort(key=lambda j: j['score'], reverse=True)
    return scored


@csrf_exempt
def find_jobs(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    resume_file = request.FILES.get('resume')
    keyword = request.POST.get('keyword', '').strip()
    location = request.POST.get('location', '').strip()

    if not resume_file:
        return JsonResponse({'error': 'Resume is required'}, status=400)

    try:
        resume_text = extract_text_from_pdf(resume_file)

        jobs = fetch_remotive_jobs(keyword, limit=15) + fetch_arbeitnow_jobs(keyword, location, limit=15)

        if not jobs:
            return JsonResponse({'jobs': []})

        scored_jobs = score_jobs_against_resume(resume_text, jobs)
        top_jobs = [j for j in scored_jobs if j['score'] >= 40][:15] or scored_jobs[:5]

        return JsonResponse({'jobs': top_jobs})

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