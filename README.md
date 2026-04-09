# Job Application Assistant

An AI-powered job application tool built with Django and OpenAI GPT-4o-mini. Upload your resume and either analyze it against a specific job description or discover which roles you're best suited for.

## Features

### Job Match Analyzer
- Paste any job description and upload your resume PDF
- Get a match score (0-100) showing how well you align with the role
- See exactly which skills you have that the job wants
- See which skills are missing so you know what to learn
- Get a fully tailored cover letter generated for that specific role and company

### Career Fit Finder
- Upload your resume once
- Get scored against 10 common developer job titles
- See which roles you're strongest for right now and which need more work
- Honest, resume-based scoring — not generic advice

## Tech Stack

- **Backend** — Django 6, Django REST Framework
- **AI** — OpenAI GPT-4o-mini
- **PDF Extraction** — pdfplumber
- **Deployment** — Railway
- **Package Management** — uv

## How It Works

1. Resume PDF is uploaded and text is extracted with pdfplumber
2. Resume text and job description are injected into a structured prompt
3. GPT-4o-mini analyzes both documents and returns a JSON response
4. Match score, skill gaps, and cover letter are rendered in the UI

This is a RAG (Retrieval Augmented Generation) pattern — the AI response is grounded in the actual content of your documents rather than relying on general knowledge alone.

## Local Setup

### Prerequisites
- Python 3.12+
- uv
- OpenAI API key

### Installation

```bash
git clone https://github.com/yourusername/job-application-assistant.git
cd job-application-assistant
uv sync
```

### Environment Variables

Create a `.env` file in the root: