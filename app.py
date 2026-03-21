import streamlit as st
import pdfplumber
import pandas as pd
import spacy
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download spaCy model if not present
if not os.path.exists("en_core_web_sm"):
    os.system("python -m spacy download en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

skills_list = pd.read_csv("skills.csv", header=None)[0].tolist()

def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

st.title("AI Resume Screening System")

job_description = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader("Upload Resumes", type="pdf", accept_multiple_files=True)

results = []

if uploaded_files and job_description:
    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        skills = extract_skills(resume_text)

        text = [resume_text, job_description]
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text)
        match = cosine_similarity(count_matrix)[0][1]
        match_percentage = round(match * 100, 2)

        results.append((file.name, match_percentage, ", ".join(skills)))

    results.sort(key=lambda x: x[1], reverse=True)

    st.write("Candidate Ranking:")
    st.table(results)