import streamlit as st
import json
from pathlib import Path

CV_FILE = Path("dataset/cv-dataset.json")

def load_cv_data():
    if CV_FILE.exists():
        try:
            with open(CV_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_cv_data(data):
    CV_FILE.parent.mkdir(exist_ok=True)
    with open(CV_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_user_cv(user_email):
    data = load_cv_data()
    for cv in data:
        if cv.get("email") == user_email:
            return cv
    return None


def profile_page():
    st.title("👤 Profilim")

    user = st.session_state.get("user", {})
    email = user.get("email")
    user_id = user.get("id")

    if not email:
        st.error("Giriş yapılmadı.")
        st.stop()

    st.markdown(f"### {user.get('name')} {user.get('surname')}")
    st.markdown(f"📧 {email}")

    all_cv = load_cv_data()
    user_cv = next((cv for cv in all_cv if cv.get("user_id") == user_id), None)

    if not user_cv:
        st.info("Henüz CV bilgisi eklenmemiş. CV bilgilerini düzenleyebilmek için önce ekleme yapın.")
        if st.button("📄 CV Bilgisi Ekle"):
            st.session_state["page"] = "cv_input"
            st.rerun()
        return

    st.markdown("---")
    st.subheader("🎓 Eğitim Bilgileri")
    edu = user_cv.get("education", {})
    level = st.text_input("Eğitim Seviyesi", edu.get("level", ""))
    university = st.text_input("Üniversite", edu.get("university", ""))
    department = st.text_input("Bölüm", edu.get("department", ""))

    st.subheader("💼 Deneyimler")
    experiences = user_cv.get("experiences", [])
    for i, exp in enumerate(experiences):
        with st.expander(f"Deneyim {i+1}: {exp['company']}"):
            exp["company"] = st.text_input(f"Şirket {i+1}", exp["company"], key=f"exp_company_{i}")
            exp["role"] = st.text_input(f"Rol {i+1}", exp["role"], key=f"exp_role_{i}")
            exp["years"] = st.text_input(f"Tecrübe Süresi {i+1}", exp["years"], key=f"exp_year_{i}")

    st.subheader("🧠 Skiller")
    skills_str = ", ".join(user_cv.get("skills", []))
    skills_input = st.text_area("Yetenekler (virgülle ayırın)", skills_str)

    st.subheader("🌍 Diller")
    langs_str = ", ".join(user_cv.get("languages", []))
    langs_input = st.text_area("Yabancı Diller (virgülle ayırın)", langs_str)

    st.subheader("📜 Sertifikalar")
    certs_str = ", ".join(user_cv.get("certificates", []))
    certs_input = st.text_area("Sertifikalar (virgülle ayırın)", certs_str)

    st.subheader("🚀 Projeler")
    projects = user_cv.get("projects", [])
    for i, p in enumerate(projects):
        with st.expander(f"Proje {i+1}: {p['title']}"):
            p["title"] = st.text_input(f"Başlık {i+1}", p["title"], key=f"proj_title_{i}")
            p["description"] = st.text_area(f"Açıklama {i+1}", p["description"], key=f"proj_desc_{i}")

    st.markdown("---")

    if st.button("💾 Değişiklikleri Kaydet"):
        updated_cv = {
            "email": email,
            "education": {"level": level, "university": university, "department": department},
            "experiences": experiences,
            "skills": [s.strip() for s in skills_input.split(",") if s.strip()],
            "languages": [l.strip() for l in langs_input.split(",") if l.strip()],
            "certificates": [c.strip() for c in certs_input.split(",") if c.strip()],
            "projects": projects
        }

        for i, cv in enumerate(all_cv):
            if cv.get("email") == email:
                all_cv[i] = updated_cv
                break

        save_cv_data(all_cv)
        st.success("✅ Profil bilgileri başarıyla güncellendi!")

    if st.button("🏠 Ana Sayfaya Dön"):
        st.session_state["page"] = "home"
        st.rerun()
