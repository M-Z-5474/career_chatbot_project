import streamlit as st
import joblib
import pandas as pd
import string
import re
import random

# Load Model and Vectorizer
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load original dataset (used for example answers)
df = pd.read_csv('career_guidance_dataset.csv')

# Clean user input
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Page Setup
st.set_page_config(page_title="Career Chatbot", page_icon="🎓")
st.title("🎓 Career Guidance Chatbot")
st.markdown("Ask something about your interests and get career recommendations!")

# Chat Input UI
user_input = st.text_input("🔍 Describe your interests or ask a question", placeholder="e.g., I love solving math problems and analyzing data")

# Add a Submit Button (arrow-style)
if st.button("➡️ Send"):
    if user_input.strip() == "":
        st.warning("Please enter something to get a career suggestion.")
    else:
        clean_input = clean_text(user_input)
        input_vec = vectorizer.transform([clean_input])
        predicted_role = model.predict(input_vec)[0]

        # Get a random answer for this role
        responses = df[df['role'] == predicted_role]['answer'].tolist()
        reply = random.choice(responses) if responses else "I suggest exploring more about this career."

        # Output
        st.success(f"🎯 Recommended Career: **{predicted_role}**")
        st.info(f"💡 {reply}")
