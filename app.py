import streamlit as st
import joblib
import pandas as pd
import string
import random

# ---------- Load Model and Data ----------
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = pd.read_csv('career_guidance_dataset.csv')

# ---------- Text Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ---------- Page Configuration ----------
st.set_page_config(page_title="Career Guidance Chatbot", page_icon="🎓", layout="centered")

# ---------- Sticky Header ----------
st.markdown("""
    <style>
    .sticky-header {
        position: sticky;
        top: 0;
        background-color: white;
        padding: 10px 0;
        z-index: 1000;
        border-bottom: 1px solid #ccc;
        text-align: center;
    }
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px 20px;
        margin-bottom: 100px;
    }
    .fixed-bottom-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 10px 25px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 999;
    }
    input[type="text"] {
        width: 85% !important;
        display: inline-block;
    }
    button[kind="primary"] {
        display: inline-block;
        margin-left: 10px;
    }
    </style>
    <div class="sticky-header">
        <h2>🎓 Career Guidance Chatbot</h2>
        <p>Get career suggestions based on your interests</p>
    </div>
""", unsafe_allow_html=True)

# ---------- Session History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Chat Scrollable Area ----------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"""
        <div style="background-color:#E7F4FF; padding:10px 15px; border-radius:18px; margin-bottom:10px; max-width:70%; margin-left:auto; text-align:right;">
            <b>You:</b><br>{message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#F5F5F5; padding:10px 15px; border-radius:18px; margin-bottom:10px; max-width:70%; text-align:left;">
            <b>Bot:</b><br>{message}
        </div>
        """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Fixed Input Bar ----------
st.markdown('<div class="fixed-bottom-input">', unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input("", placeholder="Type your interest or question here...", label_visibility="collapsed")
    with col2:
        submit_button = st.form_submit_button("➡️")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Handle Input Submission ----------
if submit_button and user_input.strip() != "":
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])
    predicted_role = model.predict(input_vec)[0]

    # Get random response for predicted role
    answers = df[df['role'] == predicted_role]['answer'].tolist()
    reply = random.choice(answers) if answers else "🤔 Sorry, I couldn't find advice for that."

    # Append to chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", f"🎯 **{predicted_role}**\n{reply}"))
