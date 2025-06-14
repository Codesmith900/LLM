import streamlit as st
import requests

st.title("NLP Preprocessing App")

text = st.text_area("Enter your text here:")

if st.button("Process Text"):
    res = requests.post("http://localhost:8000/process", json={"text": text})
    if res.status_code == 200:
        result = res.json()
        st.subheader("Tokens")
        st.write(result["tokens"])
        st.subheader("Lemmas")
        st.write(result["lemmas"])
        st.subheader("Stems")
        st.write(result["stems"])
        st.subheader("POS Tags")
        st.write(result["pos"])
        st.subheader("Named Entities")
        st.write(result["entities"])
    else:
        st.error("Error: Could not connect to backend.")
