import streamlit as st

def job_detail_page():
    st.title("📄 İş İlanı Detayı")

    job = st.session_state.get("selected_job")
    if not job:
        st.warning("Bir iş ilanı seçilmedi.")
        if st.button("⬅️ Geri Dön"):
            st.session_state["page"] = "job_matches"
            st.rerun()
        return

    # JSON yapısına uygun alanlar
    job_id = job.get("job_id", "N/A")
    job_title = job.get("job_title_clean", "Bilinmeyen Pozisyon").title()
    location = job.get("location_clean", "Belirtilmemiş")
    description = job.get("job_description_clean", "Açıklama bulunamadı.")
    mapped_title = job.get("mapped_title", "Belirtilmemiş")

    # Görsel sunum
    st.markdown(f"## {job_title}")
    st.markdown(f"🏢 **{job_id} Şirketi**")
    st.markdown(f"📍 **Lokasyon:** {location}")
    st.markdown(f"💼 **Pozisyon (Eşleştirilmiş):** {mapped_title}")

    st.markdown("---")
    st.subheader("📝 İş Tanımı")
    st.write(description)

    st.markdown("---")
    cols = st.columns([1, 6])
    with cols[0]:
        if st.button("⬅️ Listeye Dön"):
            st.session_state["page"] = "job_matches"
            st.rerun()
