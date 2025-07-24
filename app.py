import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Career Guidance Chatbot", layout="wide", page_icon="🎓")

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("career_guidance_dataset.csv")
    df.columns = [col.lower().strip() for col in df.columns]
    return df

data = load_data()

# -------------------- TF-IDF VECTOR SETUP --------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135789.png", width=100)
    st.title("🎓 CareerBot Assistant")
    st.markdown("Hi! Ask me anything about a career role.")
    st.markdown("Try:")
    st.code("What skills do I need?")
    st.code("What is the career path?")
    st.code("Do I need a degree?")
    st.markdown("---")
    st.markdown("Built with ❤️ using **Streamlit**")

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align: center;'>🔍 AI-Powered Career Guidance Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🤖 Ask about your dream role and get instant answers</p>", unsafe_allow_html=True)
st.divider()

# -------------------- EXAMPLE BUTTONS --------------------
st.markdown("**Need ideas? Click a sample question below:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧠 What does a Data Scientist do?"):
        st.session_state['example_question'] = "What does a Data Scientist do?"
with col2:
    if st.button("📈 Career growth in Data Science"):
        st.session_state['example_question'] = "What is the career growth path for a Data Scientist?"
with col3:
    if st.button("📚 Required skills for Data Scientist"):
        st.session_state['example_question'] = "What skills are required to become a Data Scientist?"

# -------------------- USER INPUT --------------------
user_input = st.text_input(
    "💬 Ask your career question below:",
    value=st.session_state.get('example_question', ''),
    placeholder="E.g., What is the job description for a Data Scientist?"
)

submit = st.button("🚀 Get Career Advice")

# -------------------- ANSWER SECTION --------------------
if submit and user_input.strip() != "":
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, X)
    top_idx = sim_scores.argmax()

    matched_row = data.iloc[top_idx]
    role = matched_row["role"]
    matched_question = matched_row["question"]
    answer = matched_row["answer"]

    # Save chat
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "question": user_input,
        "matched_q": matched_question,
        "role": role,
        "answer": answer
    })

    # Response card
    st.markdown(f"""
    <div style='background-color: #f9f9f9; padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
        <h3>🎯 Career Role: <span style='color:#4CAF50;'>{role}</span></h3>
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
