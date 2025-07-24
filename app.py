import streamlit as st
import joblib
import pandas as pd
import string
import re
import random
from sklearn.metrics.pairwise import cosine_similarity

# Load model and vectorizer
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load dataset
df = pd.read_csv('career_guidance_dataset.csv')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Page setup
st.set_page_config(page_title="Career Chatbot", page_icon="🎓")
st.title("🎓 Career Guidance Chatbot")
st.markdown("Ask about your interests or career questions to get recommendations!")

# Input UI
user_input = st.text_input("🔍 Describe your interests or ask a career question", placeholder="e.g., I love analyzing data or becoming a UX Designer")

# On send button
if st.button("➡️ Send"):
    if user_input.strip() == "":
        st.warning("Please enter a question or interest.")
    else:
        clean_input = clean_text(user_input)
        input_vec = vectorizer.transform([clean_input])
        
        # Step 1: Use ML model to predict the role
        predicted_role = model.predict(input_vec)[0]

        # Step 2: Filter dataset for that role
        role_df = df[df['role'] == predicted_role].copy()
        role_df['clean_question'] = role_df['question'].apply(clean_text)

        # Step 3: Compute similarity with role-based questions
        role_vecs = vectorizer.transform(role_df['clean_question'])
        similarity_scores = cosine_similarity(input_vec, role_vecs).flatten()
        best_index = similarity_scores.argmax()

        # Step 4: Get matched question and answer
        best_question = role_df.iloc[best_index]['question']
        best_answer = role_df.iloc[best_index]['answer']

        # Output
        st.success(f"🎯 Predicted Role: **{predicted_role}**")
        st.info(f"💬 Closest Match: *{best_question}*")
        st.info(f"💡 {best_answer}")
