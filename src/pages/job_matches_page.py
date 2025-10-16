import streamlit as st
import json
import joblib
import pandas as pd
from pathlib import Path
from pages.model_features import build_features_for_match

# --- Dosya yolları ---
JOB_POSTINGS_PATH = Path("dataset/job-postings/job-posting-dataset-all.json")
JOB_SKILLS_PATH = Path("dataset/job-postings/job-skills-all.json")
CV_DATA_PATH = Path("dataset/cv-dataset.json")
MODEL_PATH = Path("cv-job-matching/models/logistic_regression_model.pkl")

# --- Model yükleme ---
pipe = joblib.load(MODEL_PATH)


# === Yardımcı Fonksiyonlar ===
def load_json(path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def load_user_cv(user_id):
    cvs = load_json(CV_DATA_PATH)
    return next((cv for cv in cvs if cv.get("user_id") == user_id), None)


def get_job_skills(job_id):
    job_skills_data = load_json(JOB_SKILLS_PATH)
    entry = next((j for j in job_skills_data if j.get("job_id") == job_id), None)
    return entry.get("skills", []) if entry else []


def calculate_match_score(cv, job_id):
    """Model üzerinden gerçek uyum oranını hesapla."""
    if not cv:
        return 0.0

    job_skills = get_job_skills(job_id)
    cv_skills = cv.get("skills", [])

    if not job_skills or not cv_skills:
        return 0.0

    X = build_features_for_match(cv_skills, job_skills)
    score = pipe.predict_proba(X)[0][1]  # pozitif sınıf olasılığı
    return round(score * 100, 1)


# === Ana Fonksiyon ===
def job_matches_page():
    st.title("💼 İş Eşleşmeleri")

    user = st.session_state.get("user")
    if not user:
        st.error("Giriş yapılmadı.")
        st.stop()

    user_id = user.get("id")
    cv = load_user_cv(user_id)
    if not cv:
        st.warning("CV veriniz bulunamadı. Önce CV bilgilerinizi ekleyin.")
        if st.button("📄 CV Ekle"):
            st.session_state["page"] = "cv_input"
            st.rerun()
        return

    job_postings = load_json(JOB_POSTINGS_PATH)
    if not job_postings:
        st.warning("İş ilanı verisi bulunamadı.")
        return

    # Sayfalama
    jobs_per_page = 10
    total_pages = (len(job_postings) + jobs_per_page - 1) // jobs_per_page
    page_num = st.session_state.get("job_page", 1)

    start = (page_num - 1) * jobs_per_page
    end = start + jobs_per_page
    current_jobs = job_postings[start:end]

    # === Listeleme ===
    for job in current_jobs:
        job_id = job.get("job_id", "N/A")
        title = job.get("job_title_clean", "Bilinmeyen Pozisyon").title()
        location = job.get("location_clean", "Belirtilmemiş")

        match = calculate_match_score(cv, job_id)

        st.markdown(f"### {title}")
        st.write(f"{job_id} Şirketi • 📍 {location}")
        st.progress(match / 100 if match > 0 else 0.001)
        st.caption(f"🔍 Uyum Oranı: **%{match:.1f}**")

        if st.button("📄 Detay", key=f"detail_{job_id}"):
            st.session_state["selected_job"] = job
            st.session_state["page"] = "job_detail"
            st.rerun()

        st.markdown("---")

    # === Sayfa kontrol butonları ===
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Önceki", disabled=(page_num == 1)):
            st.session_state["job_page"] = page_num - 1
            st.rerun()
    with col3:
        if st.button("Sonraki ➡️", disabled=(page_num == total_pages)):
            st.session_state["job_page"] = page_num + 1
            st.rerun()

    st.markdown(f"Sayfa {page_num} / {total_pages}")

    if st.button("🏠 Ana Sayfa"):
        st.session_state["page"] = "home"
        st.rerun()
