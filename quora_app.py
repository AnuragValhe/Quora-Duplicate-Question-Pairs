import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load encoder
encoder = pickle.load(open("Saved Models/hf_encoder.pkl", "rb"))

st.title("Quora Duplicate Question Detector")

q1 = st.text_input("Enter Question 1:")
q2 = st.text_input("Enter Question 2:")

if st.button("Check Duplicate"):
    vec1 = encoder.encode([q1])[0]
    vec2 = encoder.encode([q2])[0]

    # --- Cosine similarity logic ---
    score = cosine_similarity([vec1], [vec2])[0][0]
    st.write("### Cosine Similarity Score:", round(score, 3))

    if score > 0.7:
        st.success("Duplicate")
    else:
        st.info("Not Duplicate")