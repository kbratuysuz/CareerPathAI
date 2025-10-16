import streamlit as st
import json
from pathlib import Path

from utils import get_roles_from_datasets, get_university_department_map

EDU_LEVELS = ["— Seçiniz —", "Ön Lisans (MYO)", "Lisans", "Yüksek Lisans", "Doktora"]
YEAR_RANGES = ["— Seçiniz —", "0-1 yıl", "1-3 yıl", "3-5 yıl", "5+ yıl"]
LANGUAGES = ["— Seçiniz —", "İngilizce", "Almanca", "Fransızca", "İspanyolca", "Türkçe", "Rusça"]
ROLES = ["— Seçiniz —"] + get_roles_from_datasets()

uni_dept_map = get_university_department_map()
UNIVERSITIES = ["— Seçiniz —"] + sorted(list(uni_dept_map.keys()))
DEPARTMENTS = ["— Seçiniz —"] + sorted(list(set(dept for depts in uni_dept_map.values() for dept in depts)))

def load_skills():
    p = Path("dataset/skill-list-all.json")
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            skills = json.load(f)
        return ["— Seçiniz —"] + skills
    return ["— Seçiniz —"]

ALL_SKILLS = load_skills()

def _init_state():
    ss = st.session_state
    ss.setdefault("step", 1) 
    ss.setdefault("education", {"level": None, "university": None, "department": None})
    ss.setdefault("experiences", [])
    ss.setdefault("skills", [])
    ss.setdefault("languages", [])
    ss.setdefault("certificates", [])
    ss.setdefault("projects", [])

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _reset_widget(*keys):
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = None

def _delete_button(label, key, on_click):
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button(label, key=key):
            on_click()

def step_education():
    st.header("🎓 Adım 1: Eğitim Bilgileri")
    col1, col2, col3 = st.columns(3)

    with col1:
        level = st.selectbox("Son Eğitim Seviyesi", EDU_LEVELS, key="edu_level")
    with col2:
        university = st.selectbox("Üniversite", UNIVERSITIES, key="edu_uni")
    with col3:
        department = st.selectbox("Bölüm", DEPARTMENTS, key="edu_dep")

    valid = (level and level != "— Seçiniz —" and
             university and university != "— Seçiniz —" and
             department and department != "— Seçiniz —")

    st.markdown("---")
    cols = st.columns([1, 1, 6, 1])
    with cols[0]:
        st.button("⬅️ Geri", disabled=True)
    with cols[3]:
        st.button("➡️ İleri", disabled=not valid, on_click=lambda: save_and_next(level, university, department))


def save_and_next(level, university, department):
    st.session_state["education"] = {
        "level": level,
        "university": university,
        "department": department
    }
    _go(2)

def step_experiences():
    st.header("💼 Adım 2: İş Deneyimleri")
    c1, c2, c3 = st.columns([3,2,2])
    with c1:
        company = st.text_input("Şirket Adı", key="exp_company")
    with c2:
        years = st.selectbox("Tecrübe Yılı (Aralık)", YEAR_RANGES, key="exp_years")
    with c3:
        role = st.selectbox("Rol", ROLES, key="exp_role")

    def add_exp():
        if not company or years == "— Seçiniz —" or role == "— Seçiniz —":
            st.warning("Tüm deneyim alanlarını doldurun.")
            return
        item = {"company": company.strip(), "years": years, "role": role}
        key = (_norm(item["company"]), _norm(item["role"]))
        existing = {(_norm(x["company"]), _norm(x["role"])) for x in st.session_state["experiences"]}
        if key in existing:
            st.info("Bu şirket ve rol için deneyim zaten ekli.")
            return
        st.session_state["experiences"].append(item)

        st.session_state["exp_company"] = ""
        st.session_state["exp_years"] = YEAR_RANGES[0]
        st.session_state["exp_role"] = ROLES[0]

    st.button("➕ Deneyim Ekle", on_click=add_exp)

    if st.session_state["experiences"]:
        st.subheader("Eklenen Deneyimler")
        for i, exp in enumerate(st.session_state["experiences"]):
            cols = st.columns([7,1])
            with cols[0]:
                st.markdown(f"- **{exp['company']}** | {exp['role']} ({exp['years']})")
            with cols[1]:
                if st.button("🗑️", key=f"del_exp_{i}"):
                    st.session_state["experiences"].pop(i)
                    st.session_state["rerun"] = True

    st.markdown("---")
    
    cols = st.columns([1, 1, 6, 1])
    with cols[0]:
        st.button("⬅️ Geri", on_click=go_back_to_education)
    with cols[3]:
        st.button("➡️ İleri", disabled=len(st.session_state["experiences"]) == 0, on_click=lambda: _go(3))


def go_back_to_education():
    """Eğitim adımına dönmeden önce form alanlarını eski değerlerle doldur."""
    edu = st.session_state.get("education", {})
    st.session_state["edu_level"] = edu.get("level", "— Seçiniz —")
    st.session_state["edu_uni"] = edu.get("university", "— Seçiniz —")
    st.session_state["edu_dep"] = edu.get("department", "— Seçiniz —")
    _go(1)
 
def step_skills():
    st.header("🧠 Adım 3: Yetenekler (Skills)")
    skill = st.selectbox("Bir Skill Seç", ALL_SKILLS, key="skill_pick")

    def add_skill():
        if not skill or skill == "— Seçiniz —":
            return
        exists = {_norm(s) for s in st.session_state["skills"]}
        if _norm(skill) in exists:
            st.info("Bu skill zaten ekli.")
            return
        st.session_state["skills"].append(skill)
        st.session_state["skill_pick"] = ALL_SKILLS[0]

    st.button("➕ Skill Ekle", on_click=add_skill)

    if st.session_state["skills"]:
        st.subheader("Seçilen Skiller")
        for i, s in enumerate(st.session_state["skills"]):
            cols = st.columns([7,1])
            with cols[0]:
                st.markdown(f"- {s}")
            with cols[1]:
                if st.button("🗑️", key=f"del_skill_{i}"):
                    st.session_state["skills"].pop(i)
                    st.session_state["rerun"] = True

    st.markdown("---")
    cols = st.columns([1,1,6,1])
    with cols[0]:
        st.button("⬅️ Geri", on_click=lambda: _go(2))
    with cols[3]:
        st.button("➡️ İleri", disabled=len(st.session_state["skills"]) == 0, on_click=lambda: _go(4))

def step_languages():
    st.header("🌍 Adım 4: Yabancı Diller")
    lang = st.selectbox("Bir Dil Seç", LANGUAGES, key="lang_pick")

    def add_lang():
        if not lang or lang == "— Seçiniz —":
            return
        exists = {_norm(l) for l in st.session_state["languages"]}
        if _norm(lang) in exists:
            st.info("Bu dil zaten ekli.")
            return
        st.session_state["languages"].append(lang)
        st.session_state["lang_pick"] = LANGUAGES[0]

    st.button("➕ Dil Ekle", on_click=add_lang)

    if st.session_state["languages"]:
        st.subheader("Seçilen Diller")
        for i, l in enumerate(st.session_state["languages"]):
            cols = st.columns([7,1])
            with cols[0]:
                st.markdown(f"- {l}")
            with cols[1]:
                if st.button("🗑️", key=f"del_lang_{i}"):
                    st.session_state["languages"].pop(i)
                    st.session_state["rerun"] = True

    st.markdown("---")
    cols = st.columns([1,1,6,1])
    with cols[0]:
        st.button("⬅️ Geri", on_click=lambda: _go(3))
    with cols[3]:
        st.button("➡️ İleri", disabled=len(st.session_state["languages"]) == 0, on_click=lambda: _go(5))

def step_certificates():
    st.header("📜 Adım 5: Sertifikalar")
    cert = st.text_input("Sertifika Adı", key="cert_input")

    def add_cert():
        name = (cert or "").strip()
        if not name:
            return
        exists = {_norm(c) for c in st.session_state["certificates"]}
        if _norm(name) in exists:
            st.info("Bu sertifika zaten ekli.")
            return
        st.session_state["certificates"].append(name)
        st.session_state["cert_input"] = ""

    st.button("➕ Sertifika Ekle", on_click=add_cert)

    if st.session_state["certificates"]:
        st.subheader("Eklenen Sertifikalar")
        for i, c in enumerate(st.session_state["certificates"]):
            cols = st.columns([7,1])
            with cols[0]:
                st.markdown(f"- {c}")
            with cols[1]:
                if st.button("🗑️", key=f"del_cert_{i}"):
                    st.session_state["certificates"].pop(i)
                    st.session_state["rerun"] = True

    st.markdown("---")
    cols = st.columns([1,1,6,1])
    with cols[0]:
        st.button("⬅️ Geri", on_click=lambda: _go(4))
    with cols[3]:
        st.button("➡️ İleri", disabled=len(st.session_state["certificates"]) == 0, on_click=lambda: _go(6))

def step_projects():
    st.header("🚀 Adım 6: Projeler")
    title = st.text_input("Proje Başlığı", key="proj_title")
    desc = st.text_area("Proje Açıklaması", key="proj_desc")

    def add_project():
        t = (title or "").strip()
        d = (desc or "").strip()
        if not t or not d:
            st.warning("Başlık ve açıklamayı doldurun.")
            return

        exists = {_norm(p["title"]) for p in st.session_state["projects"]}
        if _norm(t) in exists:
            st.info("Bu proje başlığı zaten ekli.")
            return
        st.session_state["projects"].append({"title": t, "description": d})
        st.session_state["proj_title"] = ""
        st.session_state["proj_desc"] = ""

    st.button("➕ Proje Ekle", on_click=add_project)

    if st.session_state["projects"]:
        st.subheader("Eklenen Projeler")
        for i, p in enumerate(st.session_state["projects"]):
            cols = st.columns([7,1])
            with cols[0]:
                st.markdown(f"**{p['title']}** – {p['description']}")
            with cols[1]:
                if st.button("🗑️", key=f"del_proj_{i}"):
                    st.session_state["projects"].pop(i)
                    st.session_state["rerun"] = True

    st.markdown("---")
    cols = st.columns([1,1,5,2])
    with cols[0]:
        st.button("⬅️ Geri", on_click=lambda: _go(5))
    with cols[3]:
        st.button("➡️ Özet", disabled=len(st.session_state["projects"]) == 0, on_click=lambda: _go(7))

def step_summary():
    st.header("✅ CV Özeti")

    user_id = st.session_state["user_id"]
    cv_id = f"cv_{user_id}"

    data = {
        "user_id": user_id,
        "cv_id": cv_id, 
        "education": st.session_state["education"],
        "experiences": st.session_state["experiences"],
        "skills": st.session_state["skills"],
        "languages": st.session_state["languages"],
        "certificates": st.session_state["certificates"],
        "projects": st.session_state["projects"],
    }

    st.markdown("### 🎓 Eğitim")
    ed = data["education"]
    st.write(f"**{ed['level']}** – {ed['university']} / {ed['department']}")

    st.markdown("### 💼 Deneyimler")
    if data["experiences"]:
        for e in data["experiences"]:
            st.markdown(f"- **{e['company']}** | {e['role']} ({e['years']})")
    else:
        st.write("_Deneyim eklenmedi._")

    st.markdown("### 🧠 Skiller")
    st.markdown(", ".join(data["skills"]) if data["skills"] else "_Henüz skill eklenmedi._")

    st.markdown("### 🌍 Diller")
    st.markdown(", ".join(data["languages"]) if data["languages"] else "_Henüz dil eklenmedi._")

    st.markdown("### 📜 Sertifikalar")
    if data["certificates"]:
        for c in data["certificates"]:
            st.markdown(f"- {c}")
    else:
        st.write("_Henüz sertifika eklenmedi._")

    st.markdown("### 🚀 Projeler")
    if data["projects"]:
        for p in data["projects"]:
            st.markdown(f"**{p['title']}** — {p['description']}")
    else:
        st.write("_Henüz proje eklenmedi._")

    st.markdown("---")
    cols = st.columns([1, 6, 1])
    with cols[0]:
        st.button("⬅️ Geri", on_click=lambda: _go(6))
    with cols[2]:
        if st.button("💾 Kaydet"):
            save_cv_data(data)


def save_cv_data(new_entry):
    """Girilen CV verilerini dataset/cv-dataset.json dosyasına kaydeder."""
    path = Path("dataset/cv-dataset.json")
    path.parent.mkdir(exist_ok=True)

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    existing.append(new_entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    st.success("✅ CV bilgileri başarıyla kaydedildi (dataset/cv-dataset.json)")

def _go(step_no: int):
    st.session_state["step"] = step_no

def _stepper_ui():
    labels = ["Eğitim", "Deneyim", "Skiller", "Diller", "Sertifikalar", "Projeler", "Özet"]
    s = st.session_state["step"]
    st.markdown(
        " ➜ ".join(
            [f"**{i+1}. {lbl}**" if i+1 == s else f"{i+1}. {lbl}" for i, lbl in enumerate(labels)]
        )
    )
    st.markdown("---")

def cv_input_wizard():
    _init_state()
    _stepper_ui()

    step = st.session_state["step"]
    if step == 1:   step_education()
    elif step == 2: step_experiences()
    elif step == 3: step_skills()
    elif step == 4: step_languages()
    elif step == 5: step_certificates()
    elif step == 6: step_projects()
    elif step == 7: step_summary()
