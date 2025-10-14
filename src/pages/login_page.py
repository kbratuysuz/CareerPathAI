import streamlit as st
import json
from pathlib import Path
import re

USER_FILE = Path("dataset/users.json")

def load_users():
    if USER_FILE.exists():
        try:
            with open(USER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def validate_email(email):
    """Basit e-posta format kontrolü"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def login_page():
    st.title("🔐 CareerPathAI Giriş")

    email = st.text_input("Mail Adresi")
    password = st.text_input("Şifre", type="password")

    if st.button("Giriş Yap"):
        users = load_users()
        for user in users:
            if user["email"] == email and user["password"] == password:
                st.session_state["user"] = user
                st.session_state["page"] = "home"
                st.success(f"Hoş geldiniz, {user['name']}!")
                st.rerun()
        st.error("❌ Geçersiz mail adresi veya şifre.")

    st.markdown("Henüz hesabınız yok mu?")
    if st.button("📝 Kaydol"):
        st.session_state["page"] = "register"
        st.rerun()
