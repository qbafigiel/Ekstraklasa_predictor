import streamlit as st

st.set_page_config(
    page_title="Ekstraklasa Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Ekstraklasa Predictor")
st.subheader("System predykcji meczów PKO BP Ekstraklasy")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Sezon", value="2025/26")

with col2:
    st.metric(label="Drużyny", value="18")

with col3:
    st.metric(label="Status", value="🟢 Działa")

st.divider()

st.info("👋 Aplikacja uruchomiona. Kolejne moduły będą dodawane stopniowo.")