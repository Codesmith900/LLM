import streamlit as st
import requests
from visualize import plot_embeddings

st.title("Word Embedding Explorer")

word = st.text_input("Enter a word:")

if st.button("Get Embedding and Neighbors"):
    res = requests.post("http://localhost:8000/embedding", json={"word": word})
    if res.status_code == 200:
        result = res.json()
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"Embedding for '{word}'")
            st.write(result["embedding"])
            st.subheader("Top Neighbors")
            neighbors = result["neighbors"]
            st.write(neighbors)

            # Prepare for visualization
            neighbor_words = [n[0] for n in neighbors]
            all_words = [word] + neighbor_words
            all_vectors = [result["embedding"]] + [requests.post("http://localhost:8000/embedding", json={"word": w}).json()["embedding"] for w in neighbor_words]
            
            plot_embeddings(all_words, all_vectors)
            st.image("embedding_plot.png")
    else:
        st.error("Could not connect to API.")
