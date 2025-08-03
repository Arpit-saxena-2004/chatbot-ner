import streamlit as st
import pickle
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy

# Load sentiment analysis model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load T5 summarization model
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load SpaCy NER
nlp = spacy.load("en_core_web_sm")

# NLP Functions
def analyze_sentiment(text):
    X = vectorizer.transform([text])
    label = sentiment_model.predict(X)[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map.get(label, str(label))

def summarize(text):
    input_text = "summarize: " + text
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = t5_model.generate(
        inputs,
        max_length=100,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Streamlit UI
st.set_page_config("NLP Chatbot", layout="centered")
st.title("🤖 NLP Chatbot - Sentiment | Summarization | NER")

# --- Chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: task selection and clear button ---
st.sidebar.title("Settings")
selected_tasks = st.sidebar.multiselect(
    "Select NLP tasks to apply:",
    ["Sentiment Analysis", "Summarization", "Named Entity Recognition"],
    default=["Sentiment Analysis", "Summarization", "Named Entity Recognition"]
)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# --- Text input or file upload ---
option = st.radio("Choose Input Mode:", ["Text", "Upload File (.txt)"])

if option == "Text":
    user_input = st.text_area("🧑 You:", placeholder="Type your message here...")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    user_input = uploaded_file.read().decode() if uploaded_file else ""

# --- Analysis ---
if st.button("Analyze") and user_input:
    st.session_state.chat_history.append(("🧑 You", user_input))

    if "Sentiment Analysis" in selected_tasks:
        sentiment = analyze_sentiment(user_input)
        st.session_state.chat_history.append(("🤖 Bot", f"**Sentiment:** {sentiment}"))

    if "Summarization" in selected_tasks:
        summary = summarize(user_input)
        st.session_state.chat_history.append(("🤖 Bot", f"**Summary:** {summary}"))

    if "Named Entity Recognition" in selected_tasks:
        entities = extract_entities(user_input)
        entity_display = ", ".join([f"`{text}` ({label})" for text, label in entities]) or "No entities found."
        st.session_state.chat_history.append(("🤖 Bot", f"**Entities:** {entity_display}"))

# --- Display chat history ---
st.divider()
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
