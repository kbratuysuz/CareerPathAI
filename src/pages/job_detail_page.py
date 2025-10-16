import streamlit as st
import json
import joblib
import pandas as pd
from pathlib import Path

# --- Dosya yolları ---
MODEL_PATH = Path("cv-job-matching/models/logistic_regression_model.pkl")
CV_DATA_PATH = Path("dataset/cv-dataset.json")
JOB_SKILLS_PATH = Path("dataset/job-postings/job-skills-all.json")
SKILL_RESOURCES_PATH = Path("dataset/skill-resources.json")

# --- Model ve veri yükleme ---
pipe = joblib.load(MODEL_PATH)

with open(SKILL_RESOURCES_PATH, "r", encoding="utf-8") as f:
    skill_resources = json.load(f)

with open(JOB_SKILLS_PATH, "r", encoding="utf-8") as f:
    all_job_skills = json.load(f)

with open(CV_DATA_PATH, "r", encoding="utf-8") as f:
    all_cvs = json.load(f)

# --- Özellik oluşturma (modelden kopya alınan) ---
def build_features(cv_skills, job_skills_with_scores):
    cv_skills = set([s.lower() for s in cv_skills])
    job_skills = set([j["skill"].lower() for j in job_skills_with_scores])

    intersection = cv_skills & job_skills
    missing = job_skills - cv_skills
    union = cv_skills | job_skills

    return {
        "skill_match_count": len(intersection),
        "skill_match_ratio": len(intersection) / len(job_skills) if job_skills else 0,
        "cv_skill_count": len(cv_skills),
        "job_skill_count": len(job_skills),
        "missing_skill_count": len(missing),
        "jaccard_similarity": len(intersection) / len(union) if union else 0
    }

def simulate_score(cv, job):
    """Bir CV ve iş ilanı için model skorunu hesapla."""
    features = build_features(cv["skills"], job["skills"])
    X = pd.DataFrame([features])
    return pipe.predict_proba(X)[0][1]


def job_detail_page():
    st.title("📄 İş İlanı Detayı")

    job = st.session_state.get("selected_job")
    user = st.session_state.get("user")

    if not job or not user:
        st.warning("İlan veya kullanıcı bilgisi bulunamadı.")
        if st.button("⬅️ Geri Dön"):
            st.session_state["page"] = "job_matches"
            st.rerun()
        return

    job_id = job.get("job_id")
    job_title = job.get("job_title_clean", "Bilinmeyen Pozisyon").title()
    location = job.get("location_clean", "Belirtilmemiş")
    description = job.get("job_description_clean", "Açıklama bulunamadı.")
    mapped_title = job.get("mapped_title", "Belirtilmemiş")

    st.markdown(f"## {job_title}")
    st.markdown(f"🏢 **{job_id} Şirketi**")
    st.markdown(f"📍 **Lokasyon:** {location}")
    st.markdown(f"💼 **Pozisyon (Eşleştirilmiş):** {mapped_title}")

    st.markdown("---")
    st.subheader("📝 İş Tanımı")
    st.write(description)
    st.markdown("---")

    # --- CV ve Job eşleşmesi ---
    user = st.session_state.get("user")
    user_id = user.get("id")
    cv = next((c for c in all_cvs if c.get("user_id") == user_id), None)
    job_skills_entry = next((j for j in all_job_skills if j.get("job_id") == job_id), None)

    if not cv or not job_skills_entry:
        st.warning("Eşleşme analizi için yeterli veri bulunamadı.")
        return

    job_skills = {j["skill"].lower(): j["score"] for j in job_skills_entry["skills"]}
    cv_skills = [s.lower() for s in cv["skills"]]
    missing_skills = {s: job_skills[s] for s in job_skills if s not in cv_skills}
    common_skills = [s for s in job_skills if s in cv_skills]

    base_score = simulate_score(cv, job_skills_entry)
    full_cv_skills = list(cv["skills"]) + list(missing_skills.keys())
    cv_full = cv.copy()
    cv_full["skills"] = full_cv_skills
    full_score = simulate_score(cv_full, job_skills_entry)
    total_effect = max(full_score - base_score, 0)

    st.markdown(f"### 🎯 Mevcut Uyum Oranı: **%{base_score * 100:.1f}**")
    st.markdown(f"### 📈 Tüm Eksik Yetenekler Tamamlansa: **%{full_score * 100:.1f}**")

    st.progress(base_score)
    st.markdown("---")

    # --- Uyuşan ve Eksik Yetenekler ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Uyuşan Yetenekler")
        if common_skills:
            for s in common_skills:
                st.markdown(f"- {s}")
        else:
            st.caption("_Henüz eşleşen yetenek bulunamadı._")

    with col2:
        st.markdown("### ⚠️ Eksik Yetenekler")
        if missing_skills:
            for s, score in missing_skills.items():
                st.markdown(f"- **{s}** (İş ilanı skoru: {score:.2f})")
        else:
            st.caption("_Eksik yetenek bulunamadı._")

    st.markdown("---")

    # --- Eksik yeteneklerin katkı analizi ---
    if total_effect > 0 and missing_skills:
        total_missing_score = sum(missing_skills.values()) or 1e-6
        weighted_results = []
        for skill, score in missing_skills.items():
            weight = score / total_missing_score
            weighted_increase = weight * total_effect
            weighted_results.append({
                "skill": skill,
                "job_score": score,
                "score_increase": weighted_increase
            })

        df = pd.DataFrame(weighted_results).sort_values(by="score_increase", ascending=False)
        st.subheader("🚀 Eksik Yetenek Katkı Analizi")
        for _, row in df.iterrows():
            st.markdown(
                f"- **{row['skill']}** → katkı: +{row['score_increase']*100:.2f}% (iş ilanı skoru: {row['job_score']:.2f})"
            )

        st.markdown("---")
        st.subheader("🧭 Gelişim Yol Haritası")

        # CSS tasarımı
        st.markdown("""
        <style>
        .roadmap-container {
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        padding: 20px 0;
        margin-bottom: 25px;
        scrollbar-width: thin;
        }

        .roadmap-step {
        min-width: 260px;
        background: #f9fafb;
        border: 2px solid #2563eb;
        border-radius: 15px;
        padding: 16px;
        margin-right: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        flex-shrink: 0;
        position: relative;
        transition: all 0.3s ease;
        }

        .roadmap-step:hover {
        transform: translateY(-4px);
        background: #eef2ff;
        }

        .roadmap-step::after {
        content: "→";
        position: absolute;
        right: -15px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 22px;
        color: #2563eb;
        }

        .roadmap-step:last-child::after {
        display: none;
        }

        .roadmap-skill {
        font-weight: 700;
        color: #1e3a8a;
        font-size: 18px;
        margin-bottom: 8px;
        }

        .roadmap-resource {
        font-size: 14px;
        margin-left: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        # HTML render
        roadmap_html = '<div class="roadmap-container">'
        for _, row in df.iterrows():
            skill_name = row["skill"].lower()
            skill_display = skill_name.title()
            roadmap_html += f'<div class="roadmap-step">'
            roadmap_html += f'<div class="roadmap-skill">⭐ {skill_display}</div>'

            if skill_name in skill_resources:
                resources = skill_resources[skill_name].get("resources", [])
                for res in resources:
                    roadmap_html += f'<div class="roadmap-resource">📘 <a href="{res["url"]}" target="_blank">{res["name"]}</a></div>'
            else:
                roadmap_html += '<div class="roadmap-resource"><em>Kaynak bulunamadı.</em></div>'

            roadmap_html += "</div>"
        roadmap_html += "</div>"

        st.markdown(roadmap_html, unsafe_allow_html=True)

    else:
        st.info("Tüm yetenekler mevcut veya eksik yeteneklerin katkısı bulunmadı.")

    st.markdown("---")
    if st.button("⬅️ Listeye Dön"):
        st.session_state["page"] = "job_matches"
        st.rerun()
