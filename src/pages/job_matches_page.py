import streamlit as st
import json
import random
from pathlib import Path
import joblib
import pandas as pd
from pages.model_features import build_features_for_match

MODEL_PATH = Path("cv-job-matching/models/logistic_regression_model.pkl")
pipe = joblib.load(MODEL_PATH)

JOB_FILE = Path("dataset/job-postings/job-posting-dataset-all.json")
CV_FILE = Path("dataset/cv-dataset.json")

def load_jobs():
    if JOB_FILE.exists():
        try:
            with open(JOB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def load_user_cv(email):
    if CV_FILE.exists():
        with open(CV_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for cv in data:
                    if cv.get("email") == email:
                        return cv
            except json.JSONDecodeError:
                pass
    return None

def calculate_match_score(cv, job):
    """Gerçek model skorunu hesapla."""
    if not cv or not job:
        return 0

    # Model giriş özelliklerini oluştur
    job_skills = job.get("skills", [])
    cv_skills = cv.get("skills", [])

    if not job_skills or not cv_skills:
        return 0

    X = build_features_for_match(cv_skills, job_skills)
    score = pipe.predict_proba(X)[:, 1]  # 0->negatif, 1->pozitif olasılık
    return round(score * 100, 1)  # yüzde

def job_matches_page():
    st.title("💼 İş Eşleşmeleri")

    user = st.session_state.get("user", {})
    if not user:
        st.error("Giriş yapılmadı.")
        st.stop()

    email = user.get("email")
    user_cv = load_user_cv(email)

    jobs = load_jobs()
    if not jobs:
        st.warning("İş ilanı verisi bulunamadı.")
        return

    # Sayfalama
    jobs_per_page = 10
    total_pages = (len(jobs) + jobs_per_page - 1) // jobs_per_page
    page_num = st.session_state.get("job_page", 1)

    start = (page_num - 1) * jobs_per_page
    end = start + jobs_per_page
    current_jobs = jobs[start:end]

    # Listeleme
    for job in current_jobs:
        job_id = job.get("job_id", "N/A")
        title = job.get("job_title_clean", "Bilinmeyen Pozisyon")
        company = f"{job_id} Şirketi"
        match = calculate_match_score(user_cv, job)

        st.markdown(f"### {title}")
        st.write(company)
        st.progress(match / 100)
        st.caption(f"🔍 Uyum Oranı: **%{match}**")

        if st.button("📄 Detay", key=f"detail_{job_id}"):
            st.session_state["selected_job"] = job
            st.session_state["page"] = "job_detail"
            st.rerun()

        st.markdown("---")

    # Sayfa kontrol butonları
    cols = st.columns(3)
    with cols[0]:
        if st.button("⬅️ Önceki", disabled=(page_num == 1)):
            st.session_state["job_page"] = page_num - 1
            st.rerun()
    with cols[2]:
        if st.button("Sonraki ➡️", disabled=(page_num == total_pages)):
            st.session_state["job_page"] = page_num + 1
            st.rerun()

    st.markdown(f"Sayfa {page_num} / {total_pages}")

    if st.button("🏠 Ana Sayfa"):
        st.session_state["page"] = "home"
        st.rerun()
