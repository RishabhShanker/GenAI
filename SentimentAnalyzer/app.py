# app.py
import streamlit as st
from src.pipeline import run_pipeline

st.set_page_config(page_title="Market Sentiment Analyzer", page_icon="📈", layout="centered")
st.title("📈 Market Sentiment Analyzer")

company = st.text_input("Company name", "Google")
if st.button("Analyze"):
    with st.spinner("Analyzing…"):
        try:
            out = run_pipeline(company)
            st.subheader("Result JSON")
            st.json(out)
            if out.get("newsdesc"):
                st.subheader("Recent News")
                st.markdown(out["newsdesc"])
        except Exception as e:
            st.error(f"{e}")
