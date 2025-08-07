import os
import re
import string
import pytesseract
from PIL import Image
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
import spacy
from collections import Counter, defaultdict

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

CUSTOM_STOPWORDS = set(["the", "and", "in", "of", "to", "a", "for", "with", "on", "at", "by", "an", "be"])

# Massive keyword bank (can grow dynamically)
JOB_FIELDS = defaultdict(set)
JOB_FIELDS.update({
    "Software Development": set([
        "python", "java", "c++", "c#", "javascript", "typescript", "node", "react", "django", "flask",
        "backend", "frontend", "developer", "programming", "software", "git", "api", "sql", "linux",
        "bash", "docker", "kubernetes", "json", "xml", "agile", "scrum", "oop", "rest", "graphql",
        "unit test", "integration", "ci", "cd", "cloud", "aws", "azure", "gcp", "jira", "version control"
    ]),
    "Graphic Design": set([
        "illustrator", "photoshop", "figma", "indesign", "adobe", "design", "logo", "poster", "branding",
        "typography", "color", "ux", "ui", "mockup", "layout", "sketch", "animation", "visual", "portfolio",
        "coreldraw", "wireframe", "responsive", "creative suite", "adobe xd", "procreate", "pixel", "vector"
    ]),
    "Marketing": set([
        "seo", "sem", "google ads", "facebook ads", "branding", "campaign", "social media", "analytics",
        "email marketing", "market research", "content", "strategy", "influencer", "blog", "engagement",
        "roi", "kpi", "conversion", "click through", "instagram", "linkedin", "marketing automation",
        "growth", "advertising", "branding", "newsletter"
    ]),
    "Education": set([
        "teacher", "teaching", "lecturer", "education", "bachelor", "msc", "phd", "school", "tutor",
        "learning", "curriculum", "assignment", "classroom", "students", "university", "online class",
        "homeschool", "syllabus", "pedagogy", "educator", "lesson plan", "education policy"
    ]),
    "Business": set([
        "finance", "sales", "account", "project", "manage", "business", "operation", "crm",
        "strategy", "leadership", "budget", "revenue", "kpi", "plan", "negotiation", "pnl",
        "stakeholder", "reporting", "business intelligence", "excel", "forecast", "planning",
        "market", "competition", "merger", "startup", "scalability"
    ]),
    "Data Science": set([
        "machine learning", "data", "pandas", "numpy", "statistics", "data science", "deep learning",
        "tensorflow", "keras", "matplotlib", "visualization", "analysis", "model", "ai", "ml", "dataset",
        "jupyter", "classification", "regression", "clustering", "predictive", "big data", "hadoop",
        "spark", "etl", "data wrangling", "dashboards", "notebook"
    ]),
    "Cybersecurity": set([
        "network", "firewall", "security", "cyber", "threat", "pentest", "vulnerability",
        "encryption", "malware", "ethical hacking", "forensics", "iso", "compliance", "antivirus",
        "phishing", "zero-day", "ddos", "cyber law", "ransomware", "red teaming", "siem"
    ]),
    "Healthcare": set([
        "doctor", "nurse", "clinical", "patient", "medical", "surgery", "hospital", "healthcare",
        "treatment", "diagnosis", "lab", "report", "therapy", "infection", "care", "medication",
        "x-ray", "ct scan", "prescription", "symptom", "blood test", "icu", "pharmacy", "first aid"
    ])
})

def predict_level(text):
    text = text.lower()
    if any(word in text for word in ["intern", "fresher", "entry"]):
        return "Internship / Entry"
    elif any(word in text for word in ["senior", "lead", "manager"]):
        return "Professional / Senior"
    else:
        return "Mid-level"

def predict_job_type(keywords):
    scores = {field: 0 for field in JOB_FIELDS}
    for word in keywords:
        for field in JOB_FIELDS:
            if word.lower() in JOB_FIELDS[field]:
                scores[field] += 1
    best_match = max(scores, key=scores.get)
    return best_match if scores[best_match] > 0 else "Unknown"

def predict_pdf_type(text):
    text = text.lower()
    if any(x in text for x in ["cv", "resume", "skills", "experience"]):
        return "Resume"
    elif any(x in text for x in ["invoice", "receipt", "amount", "bill"]):
        return "Invoice"
    elif any(x in text for x in ["report", "summary", "introduction"]):
        return "Report / Article"
    else:
        return "Other / Unknown"

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext.endswith(".pdf"):
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif ext.endswith(('.png', '.jpg', '.jpeg')):
        return pytesseract.image_to_string(Image.open(file_path))
    return ""

def extract_keywords(text):
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.text.lower() not in CUSTOM_STOPWORDS]
    return dict(Counter(words).most_common(30))

def learn_keywords(keywords):
    for word in keywords:
        if word in CUSTOM_STOPWORDS:
            continue
        # Assign to unknown if doesn't match
        matched = False
        for field in JOB_FIELDS:
            if word.lower() in JOB_FIELDS[field]:
                matched = True
                break
        if not matched:
            JOB_FIELDS["Unknown"].add(word)

@app.route('/', methods=['GET', 'POST'])
def home():
    extracted = {}
    job_type = ""
    job_level = ""
    pdf_type = ""

    if request.method == 'POST':
        file = request.files['resume']
        if file:
            file_path = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(file_path)

            text = extract_text(file_path)
            extracted = extract_keywords(text)
            learn_keywords(extracted.keys())
            job_type = predict_job_type(extracted.keys())
            job_level = predict_level(text)
            pdf_type = predict_pdf_type(text)

    return render_template('index.html', extracted=extracted, job_type=job_type, job_level=job_level, pdf_type=pdf_type)

if __name__ == '__main__':
    app.run(debug=True)
