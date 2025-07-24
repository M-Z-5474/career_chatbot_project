import streamlit as st
import pandas as pd
import string
import joblib
import random
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Career Chatbot", layout="wide", page_icon="🎓")

# -------------------- LOAD MODEL & VECTORIZER --------------------
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("career_guidance_dataset.csv")
    df.columns = [col.lower().strip() for col in df.columns]
    return df

data = load_data()

# -------------------- CLEAN TEXT --------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135789.png", width=100)
    st.title("🎓 CareerBot Assistant")
    st.markdown("Hi! Ask me anything about your career interest.")
    st.markdown("Try:")
    st.code("What is the job growth?")
    st.code("What skills do I need?")
    st.code("Do I need a degree?")
    st.markdown("---")
    st.markdown("Built with ❤️ using **ML + Streamlit**")

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align: center;'>🔍 AI-Powered Career Guidance Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🤖 Get smart advice based on your career interests</p>", unsafe_allow_html=True)
st.divider()

# -------------------- EXAMPLE BUTTONS --------------------
st.markdown("**Not sure? Try a sample question:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧠 What does a UX Designer do?"):
        st.session_state['example_question'] = "What does a UX Designer do?"
with col2:
    if st.button("📈 Growth in Product Management"):
        st.session_state['example_question'] = "What is the career growth path for a Product Manager?"
with col3:
    if st.button("📚 Skills needed for Data Scientist"):
        st.session_state['example_question'] = "What skills are required to become a Data Scientist?"

# -------------------- USER INPUT --------------------
user_input = st.text_input(
    "💬 Ask your career-related question:",
    value=st.session_state.get('example_question', ''),
    placeholder="E.g., What is the job description for a UX Designer?"
)

submit = st.button("🚀 Get Career Advice")

# -------------------- PROCESS AND RESPOND --------------------
if submit and user_input.strip() != "":
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])

    # 1. Predict role via ML model
    predicted_role = model.predict(input_vec)[0]

    # 2. Filter data to only that role
    role_df = data[data['role'] == predicted_role].copy()
    role_df['clean_question'] = role_df['question'].apply(clean_text)

    # 3. Compute similarity to find closest Q
    role_vecs = vectorizer.transform(role_df['clean_question'])
    sim_scores = cosine_similarity(input_vec, role_vecs).flatten()
    best_index = sim_scores.argmax()

    matched_row = role_df.iloc[best_index]
    matched_question = matched_row['question']
    answer = matched_row['answer']

    # Save chat history
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "question": user_input,
        "matched_q": matched_question,
        "role": predicted_role,
        "answer": answer
    })

    # Show output
    st.markdown(f"""
    <div style='background-color: #f9f9f9; padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
        <h3>🎯 Career Role: <span style='color:#4CAF50;'>{predicted_role}</span></h3>
        <p><b>Matched Question:</b> {matched_question}</p>
        <p><b>💬 Answer:</b> {answer}</p>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

# -------------------- CHAT HISTORY --------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Chat History")
    for chat in reversed(st.session_state.history):
        st.markdown(f"""
        <div style="border:1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 10px;">
            <p>🕒 {chat['time']}</p>
            <p><b>🙋 You:</b> {chat['question']}</p>
            <p><b>🔁 Matched:</b> {chat['matched_q']}</p>
            <p><b>🤖 Bot:</b> <i>{chat['role']}</i> ➤ {chat['answer']}</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------- CLEANUP --------------------
if "example_question" in st.session_state:
    del st.session_state["example_question"]
