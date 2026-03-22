import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Skills list
skills_list = [
    "Python", "SQL", "Java", "JavaScript", "React", "Node.js",
    "Machine Learning", "Deep Learning", "NLP", "Data Analysis",
    "Excel", "Power BI", "Tableau", "Communication", "Leadership",
    "Project Management", "Agile", "Scrum", "AWS", "Docker",
    "Kubernetes", "Git", "REST API", "MongoDB", "PostgreSQL",
    "C++", "C#", "TypeScript", "Angular", "Vue.js", "Django",
    "Flask", "FastAPI", "TensorFlow", "PyTorch", "Pandas", "NumPy"
]

# Extract matched skills from text
def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

# ── Streamlit UI ──────────────────────────────────────────────
st.set_page_config(page_title="AI Resume Screener", page_icon="📄")
st.title("📄 AI Resume Screening System")
st.markdown("Upload multiple resumes and paste a job description to rank candidates.")

job_description = st.text_area("📋 Paste Job Description Here", height=200)
uploaded_files = st.file_uploader("📁 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🔍 Screen Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        results = []

        with st.spinner("Analyzing resumes..."):
            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file)

                if not resume_text.strip():
                    st.warning(f"Could not extract text from {file.name}. Skipping.")
                    continue

                skills = extract_skills(resume_text)

                texts = [resume_text, job_description]
                cv = CountVectorizer()
                count_matrix = cv.fit_transform(texts)
                match = cosine_similarity(count_matrix)[0][1]
                match_percentage = round(match * 100, 2)

                results.append({
                    "Resume": file.name,
                    "Match %": match_percentage,
                    "Skills Found": ", ".join(skills) if skills else "None detected"
                })

        if results:
            results.sort(key=lambda x: x["Match %"], reverse=True)

            st.success(f"✅ Screened {len(results)} resume(s) successfully!")
            st.subheader("🏆 Candidate Ranking")

            df = pd.DataFrame(results)
            df.index += 1
            st.dataframe(df, use_container_width=True)

            st.info(f"🥇 Top Candidate: **{results[0]['Resume']}** with {results[0]['Match %']}% match")
