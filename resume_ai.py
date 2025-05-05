# resume_ai.py

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
text_gen = pipeline("text-generation", model="gpt2")


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
    try:
        # Generate suggestions
        suggestions = text_gen(prompt, max_new_tokens=150)
        
        # Check if the response contains the expected data
        if suggestions and isinstance(suggestions, list) and 'generated_text' in suggestions[0]:
            return suggestions[0]['generated_text'].split("\n")[0]
        else:
            return "No suggestions generated. Please check the model's response."
    except Exception as e:
        print(f"Error in text generation: {e}")
        return "There was an issue generating suggestions."