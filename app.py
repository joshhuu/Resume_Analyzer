# app.py

import streamlit as st
from resume_ai import extract_text_from_pdf, clean_text, extract_entities, get_keywords, compute_similarity, generate_suggestions

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
