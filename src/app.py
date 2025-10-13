import streamlit as st
from pages.cv_input_wizard import cv_input_wizard

st.set_page_config(page_title="CareerPathAI", layout="wide")

# --- Ana Sayfa ---
def home_page():
    st.title("💼 CareerPathAI")
    st.markdown("### Yapay Zeka Destekli Kariyer Uyumluluk ve Gelişim Analiz Sistemi")
    st.write("""
    Bu sistem, CV’nizdeki becerileri analiz eder, iş ilanlarıyla olan eşleşmenizi hesaplar ve eksik yetkinlikleriniz için kişisel bir gelişim yol haritası sunar.
    """)

    if st.button("📄 CV Yükle"):
        st.session_state["page"] = "cv_input"

# --- Sayfa Yönlendirme ---
if "page" not in st.session_state:
    st.session_state["page"] = "home"

if st.session_state["page"] == "home":
    home_page()
elif st.session_state["page"] == "cv_input":
    cv_input_wizard()
