import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page title
st.title("Text Embedding and Similarity App")

st.write("Enter a sentence to find the most similar text from the dataset.")

# Load dataset
df = pd.read_csv("data/text_generation_results.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load saved embeddings
embeddings = np.load("data/embeddings.npy")

# Display dataset
st.subheader("Text Dataset")
st.dataframe(df)

# User input
user_text = st.text_input("Enter your text:")

if st.button("Find Similar Text"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:
        # Convert input text into embedding
        user_embedding = model.encode([user_text])

        # Calculate cosine similarity
        similarities = cosine_similarity(
            user_embedding,
            embeddings
        )[0]

        # Find most similar text
        best_index = similarities.argmax()

        best_text = df.iloc[best_index]["Generated_Text"]
        best_score = similarities[best_index]

        # Display result
        st.subheader("Most Similar Text")

        st.write(best_text)

        st.write(
            f"Cosine Similarity: {best_score:.4f}"
        )