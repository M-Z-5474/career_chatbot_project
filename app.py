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
st.set_page_config(page_title="Career Guidance Chatbot", page_icon="🎓", layout="wide")

# ---------- Custom CSS for Sticky Header, Scrollable Chat, Fixed Input ----------
st.markdown("""
    <style>
    /* Sticky header */
    .sticky-header {
        position: sticky;
        top: 0;
        background-color: white;
        padding: 15px 0;
        z-index: 1000;
        border-bottom: 1px solid #ccc;
        text-align: center;
    }

    /* Scrollable chat area */
    .chat-container {
        height: 65vh;
        overflow-y: auto;
        padding: 15px 25px;
        margin-bottom: 80px;
        background-color: #f9f9f9;
        border-radius: 8px;
    }

    /* Fixed input area */
    .fixed-bottom-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 10px 25px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.07);
        z-index: 999;
    }

    input[type="text"] {
        width: 100%;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ccc;
    }

    button[kind="primary"] {
        margin-top: 10px;
    }

    /* Hide Streamlit default header and footer */
    header, footer {visibility: hidden;}
    </style>

    <div class="sticky-header">
        <h2>🎓 Career Guidance Chatbot</h2>
        <p style="margin: 0; font-size: 16px; color: gray;">Get career suggestions based on your interests</p>
    </div>
""", unsafe_allow_html=True)

# ---------- Session State ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Scrollable Chat Display ----------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"""
            <div style="background-color:#E3F2FD; padding:10px 15px; border-radius:15px; margin-bottom:10px; max-width:70%; margin-left:auto; text-align:right;">
                <b>You:</b><br>{message}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color:#F1F1F1; padding:10px 15px; border-radius:15px; margin-bottom:10px; max-width:70%; text-align:left;">
                <b>Bot:</b><br>{message}
            </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Fixed Input Box ----------
st.markdown('<div class="fixed-bottom-input">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="Type your interest or question...", label_visibility="collapsed")
    submit_button = st.form_submit_button("Send")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prediction and Chat Handling ----------
if submit_button and user_input.strip() != "":
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])
    predicted_role = model.predict(input_vec)[0]

    answers = df[df['role'] == predicted_role]['answer'].tolist()
    reply = random.choice(answers) if answers else "🤔 Sorry, I couldn't find advice for that."

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", f"🎯 **{predicted_role}**\n{reply}"))
