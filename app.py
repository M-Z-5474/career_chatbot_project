import streamlit as st
import joblib
import pandas as pd
import string
import re
import random

# ------------------ Setup ------------------

# Load model/vectorizer
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = pd.read_csv('career_guidance_dataset.csv')

# Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ------------------ UI Config ------------------

st.set_page_config(page_title="Career Chatbot", page_icon="🎓", layout="centered")
st.title("🎓 Career Guidance Chatbot")
st.markdown("Type your interests or a question to receive a career suggestion.")

# ------------------ Session State for Chat ------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Input Form ------------------

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Your Message", placeholder="e.g., I love solving math problems...", label_visibility="collapsed")
    submit_button = st.form_submit_button("➡️")

# ------------------ On Submit ------------------

if submit_button and user_input.strip() != "":
    # Clean and predict
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])
    predicted_role = model.predict(input_vec)[0]

    # Get response
    answers = df[df['role'] == predicted_role]['answer'].tolist()
    reply = random.choice(answers) if answers else "Sorry, I couldn't find advice for that career."

    # Append messages
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", f"🎯 **{predicted_role}**\n💡 {reply}"))

# ------------------ Display Chat ------------------

for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"""
        <div style="background-color:#DCF8C6; padding:10px 15px; border-radius:12px; margin-bottom:8px; width:fit-content; max-width:80%; align-self:flex-end; margin-left:auto;">
            <b>You:</b><br>{message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#F1F0F0; padding:10px 15px; border-radius:12px; margin-bottom:8px; width:fit-content; max-width:80%;">
            <b>Bot:</b><br>{message}
        </div>
        """, unsafe_allow_html=True)

# ------------------ Optional: Clear Chat Button ------------------

st.markdown("---")
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
