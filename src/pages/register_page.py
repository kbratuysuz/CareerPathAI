import streamlit as st
import json
import re
from pathlib import Path

USER_FILE = Path("dataset/users.json")

def load_users():
    if USER_FILE.exists():
        try:
            with open(USER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_users(users):
    USER_FILE.parent.mkdir(exist_ok=True)
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def validate_email(email):
    """Basit e-posta format kontrolü"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def register_page():
    st.title("📝 CareerPathAI - Kaydol")

    name = st.text_input("Ad")
    surname = st.text_input("Soyad")
    email = st.text_input("Mail Adresi")
    password = st.text_input("Şifre", type="password")

    if st.button("Kaydol"):
        if not name or not surname or not email or not password:
            st.warning("Lütfen tüm alanları doldurun.")
            return

        if not validate_email(email):
            st.warning("Lütfen geçerli bir e-posta adresi girin.")
            return

        users = load_users()
        if any(u["email"] == email for u in users):
            st.error("❌ Bu e-posta adresi zaten kayıtlı.")
            return

        new_id = users[-1]["id"] + 1 if users else 1

        new_user = {
            "id": new_id,
            "name": name,
            "surname": surname,
            "email": email,
            "password": password
        }
        users.append(new_user)
        save_users(users)

        st.success("✅ Kayıt başarıyla oluşturuldu. Giriş sayfasına yönlendiriliyorsunuz...")
        st.session_state["page"] = "login"
        st.rerun()

    if st.button("🔙 Geri Dön"):
        st.session_state["page"] = "login"
        st.rerun()
