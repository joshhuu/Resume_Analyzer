# resume_ai_analyzer.py

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load NER model
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

# Load text generation model
text_gen = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)

def extract_entities(text):
    return ner_pipeline(text)

def get_keywords(entities):
    return list(set([ent['word'] for ent in entities if ent['entity_group'] in ['ORG', 'SKILL', 'JOB']]))

def compute_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def generate_suggestions(resume_text, job_desc):
    prompt = f"Given this resume: {resume_text}\nand this job description: {job_desc}, suggest improvements in 100 words."
    suggestions = text_gen(prompt, max_new_tokens=150)[0]['generated_text']
    return suggestions.split("\n")[0]

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer")
st.title("🤖 AI Resume Analyzer & Enhancer")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste the job description")

if uploaded_file and job_desc:
    resume_text = clean_text(extract_text_from_pdf(uploaded_file))
    entities = extract_entities(resume_text)
    keywords = get_keywords(entities)
    sim_score = compute_similarity(resume_text, job_desc)
    suggestions = generate_suggestions(resume_text, job_desc)

    st.subheader("🔍 Extracted Keywords")
    st.write(", ".join(keywords))

    st.subheader("📈 Relevance Score")
    st.metric(label="Match %", value=f"{round(sim_score*100, 2)}%")

    st.subheader("🧠 Suggestions")
    st.write(suggestions)
else:
    st.info("Please upload a resume and job description to get started.")
