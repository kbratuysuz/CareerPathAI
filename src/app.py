import streamlit as st
from pages.cv_input_wizard import cv_input_wizard
from pages.login_page import login_page
from pages.register_page import register_page
from pages.profile_page import profile_page
from pages.job_matches_page import job_matches_page
from pages.job_detail_page import job_detail_page

st.set_page_config(page_title="CareerPathAI", layout="wide")

def home_page():
    st.title("💼 CareerPathAI")
    st.markdown("### Yapay Zeka Destekli Kariyer Uyumluluk ve Gelişim Analiz Sistemi")

    user = st.session_state.get("user", {})
    st.write(f"👋 Hoş geldiniz **{user.get('name', 'Kullanıcı')} {user.get('surname', '')}**")

    st.write("""
    Bu sistem, CV’nizdeki becerileri analiz eder, iş ilanlarıyla olan eşleşmenizi hesaplar 
    ve eksik yetkinlikleriniz için kişisel bir gelişim yol haritası sunar.
    """)

    if st.button("📄 CV Yükle"):
        st.session_state["page"] = "cv_input"

    if st.button("👤 Profilim"):
        st.session_state["page"] = "profile"
        st.rerun()
        
    if st.button("🎯 İş Eşleşmelerim"):
        st.session_state["page"] = "job_matches"
        st.rerun()

    if st.button("🚪 Çıkış Yap"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.session_state["rerun"] = True
            
if "page" not in st.session_state:
    st.session_state["page"] = "login"

page = st.session_state["page"]

if page == "login":
    login_page()
elif page == "register":
    register_page()
elif page == "home":
    home_page()
elif page == "cv_input":
    cv_input_wizard()
elif page == "profile":
    profile_page()
elif page == "job_matches":
    job_matches_page()
elif page == "job_detail":
    job_detail_page()