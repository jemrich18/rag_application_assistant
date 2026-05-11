import pytest
import json
from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch, MagicMock
import io


class IndexViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_returns_200(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_index_uses_correct_template(self):
        response = self.client.get('/')
        self.assertTemplateUsed(response, 'analyzer/index.html')


class AnalyzeViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_analyze_requires_post(self):
        response = self.client.get('/analyze/')
        self.assertEqual(response.status_code, 405)

    def test_analyze_requires_resume_and_job_description(self):
        response = self.client.post('/analyze/', {
            'job_description': 'Python developer needed'
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn('error', data)

    def test_analyze_requires_resume(self):
        response = self.client.post('/analyze/', {})
        self.assertEqual(response.status_code, 400)

    @patch('analyzer.views.get_embedding')
    @patch('analyzer.views.analyze_with_ai')
    @patch('analyzer.views.extract_text_from_pdf')
    def test_analyze_returns_result(
        self, mock_extract, mock_analyze, mock_embedding
    ):
        mock_extract.return_value = "Python developer with Django experience"
        mock_embedding.return_value = [0.1] * 1536
        mock_analyze.return_value = {
            "score": 85,
            "present_skills": ["Python", "Django"],
            "missing_skills": ["AWS"],
            "cover_letter": "Dear Hiring Manager..."
        }

        fake_pdf = io.BytesIO(b"fake pdf content")
        fake_pdf.name = "resume.pdf"

        response = self.client.post('/analyze/', {
            'job_description': 'Python Django developer needed',
            'resume': fake_pdf
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('score', data)
        self.assertIn('present_skills', data)
        self.assertIn('missing_skills', data)
        self.assertIn('cover_letter', data)


class CareerFitViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_career_fit_requires_post(self):
        response = self.client.get('/fit/')
        self.assertEqual(response.status_code, 405)

    def test_career_fit_requires_resume(self):
        response = self.client.post('/fit/', {})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn('error', data)

    @patch('analyzer.views.career_fit_with_ai')
    @patch('analyzer.views.extract_text_from_pdf')
    def test_career_fit_returns_roles(self, mock_extract, mock_career_fit):
        mock_extract.return_value = "Python developer with Django experience"
        mock_career_fit.return_value = {
            "roles": [
                {
                    "title": "Junior Django Developer",
                    "match": 90,
                    "notes": "Strong Django background"
                }
            ]
        }

        fake_pdf = io.BytesIO(b"fake pdf content")
        fake_pdf.name = "resume.pdf"

        response = self.client.post('/fit/', {
            'resume': fake_pdf
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('roles', data)


class ExtractTextTest(TestCase):
    @patch('analyzer.views.pdfplumber.open')
    def test_extract_text_from_pdf(self, mock_pdfplumber):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample resume text"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf

        from analyzer.views import extract_text_from_pdf
        fake_pdf = io.BytesIO(b"fake pdf content")
        result = extract_text_from_pdf(fake_pdf)

        self.assertIn("Sample resume text", result)