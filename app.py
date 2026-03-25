import streamlit as st
import json
import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

# ─── Load Environment Variables ─────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# ─── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Learning Advisor",
    page_icon="🎓",
    layout="centered"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.metric-card {
    background: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin: 5px;
}
</style>
""", unsafe_allow_html=True)

# ─── Load FAQ Data ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_faq_data():
    with open("faq_data.json", "r") as f:
        data = json.load(f)
    return data["faqs"]

# ─── Load Semantic Model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# ─── Build Embeddings ─────────────────────────────────────────────────────────────
@st.cache_resource
def build_embeddings(_faqs, _model):
    questions  = [faq["question"] + " " + " ".join(faq["keywords"]) for faq in _faqs]
    embeddings = _model.encode(questions)
    return embeddings

faqs       = load_faq_data()
model      = load_model()
embeddings = build_embeddings(faqs, model)

# ─── Groq Client ─────────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

# ─── RAG: Retrieve Top-K ─────────────────────────────────────────────────────────
def retrieve_top_k(user_input, k=3, threshold=0.2):
    user_embedding = model.encode([user_input])
    similarities   = cosine_similarity(user_embedding, embeddings).flatten()
    top_indices    = np.argsort(similarities)[::-1][:k]
    results = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            results.append({
                "question": faqs[idx]["question"],
                "answer":   faqs[idx]["answer"],
                "score":    float(similarities[idx])
            })
    return results

# ─── RAG: Generate with LLM ──────────────────────────────────────────────────────
def generate_answer(user_input, retrieved_docs, chat_history):
    # Build context from retrieved docs
    context = "\n\n".join([
        f"Q: {doc['question']}\nA: {doc['answer']}"
        for doc in retrieved_docs
    ])

    # Build conversation history
    history_text = ""
    if len(chat_history) > 0:
        recent = chat_history[-4:]
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in recent
        ])

    system_prompt = f"""You are an expert AI Learning Advisor specializing in Machine Learning, Deep Learning, NLP, and Data Science.

Your role is to:
1. Answer questions clearly and concisely
2. Use the provided context to give accurate answers
3. Remember the conversation history for context
4. Suggest related topics when relevant
5. Encourage learning with a positive tone

CONTEXT FROM KNOWLEDGE BASE:
{context}

RECENT CONVERSATION:
{history_text}

Guidelines:
- Be concise but comprehensive
- Use bullet points for clarity when needed
- End with a relevant follow-up suggestion
- If the context doesn't fully answer the question, use your knowledge but mention it
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_input}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

# ─── Evaluation Metrics ───────────────────────────────────────────────────────────
def evaluate_retrieval():
    test_cases = [
        {"query": "What is ML?",                    "expected": "machine learning"},
        {"query": "explain overfitting",             "expected": "overfitting"},
        {"query": "how does neural network work",    "expected": "neural network"},
        {"query": "what is nlp",                     "expected": "nlp"},
        {"query": "difference supervised unsupervised", "expected": "supervised"},
    ]
    correct = 0
    for tc in test_cases:
        results = retrieve_top_k(tc["query"], k=3)
        for r in results:
            if tc["expected"].lower() in r["question"].lower():
                correct += 1
                break
    return round((correct / len(test_cases)) * 100, 1)

# ─── Roadmap ─────────────────────────────────────────────────────────────────────
def get_roadmap(topic):
    roadmaps = {
        "Machine Learning": """
**🗺️ ML Roadmap:**
1. Python & Math basics
2. Data preprocessing & EDA
3. Supervised Learning
4. Unsupervised Learning
5. Model Evaluation & Tuning
6. Advanced: XGBoost, Ensemble Methods
""",
        "Deep Learning": """
**🗺️ Deep Learning Roadmap:**
1. Neural Networks basics
2. Backpropagation & Optimization
3. CNNs for Computer Vision
4. RNNs & LSTMs
5. Transfer Learning
6. Advanced: GANs, Transformers
""",
        "NLP": """
**🗺️ NLP Roadmap:**
1. Text Preprocessing
2. TF-IDF & Word Embeddings
3. Sentiment Analysis
4. Named Entity Recognition
5. Transformers & BERT
6. Advanced: GPT, LLMs, RAG
""",
        "Data Science": """
**🗺️ Data Science Roadmap:**
1. Python & SQL
2. Pandas & NumPy
3. Data Visualization
4. EDA & Feature Engineering
5. ML Basics
6. Advanced: MLOps, Cloud
"""
    }
    return roadmaps.get(topic, None)

# ─── UI ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 AI Learning Advisor</h1>
    <p>Powered by RAG + Groq LLM | Your Personal AI & Data Science Expert</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📚 Topics:")
    st.markdown("""
    - 🤖 Machine Learning
    - 🧠 Deep Learning
    - 🔤 NLP
    - 📊 Data Science & EDA
    - 📈 Model Evaluation
    - 🛠️ Tools & Libraries
    - ☁️ MLOps & Deployment
    """)
    st.markdown("---")
    st.markdown(f"**📝 Knowledge Base: {len(faqs)} Q&A**")
    st.markdown("---")

    # Evaluation Metrics
    st.header("📊 System Metrics:")
    if st.button("🔍 Run Evaluation"):
        with st.spinner("Evaluating..."):
            accuracy = evaluate_retrieval()
        st.success(f"✅ Top-3 Retrieval Accuracy: **{accuracy}%**")
    st.markdown("---")

    # Roadmap Section
    st.header("🗺️ Learning Roadmap:")
    roadmap_topic = st.selectbox(
        "Choose a topic:",
        ["", "Machine Learning", "Deep Learning", "NLP", "Data Science"]
    )
    if roadmap_topic:
        roadmap = get_roadmap(roadmap_topic)
        if roadmap:
            st.markdown(roadmap)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages      = []
        st.session_state.last_question = None
        st.session_state.question_count = 0
        st.rerun()

# ─── Session State ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages       = []
    st.session_state.last_question  = None
    st.session_state.question_count = 0
    st.session_state.messages.append({
        "role":    "assistant",
        "content": "👋 Hello! I'm your **AI Learning Advisor** powered by **RAG + Groq LLM** 🎓\n\n🧠 I use **Semantic Search** to find relevant knowledge and **LLM** to generate personalized answers!\n\nAsk me anything about ML, Deep Learning, NLP, or Data Science!"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Chat Input ───────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask your question here..."):
    st.session_state.question_count += 1

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Typing effect
        with st.spinner("🔍 Searching knowledge base..."):
            retrieved_docs = retrieve_top_k(user_input, k=3)
            time.sleep(0.5)

        if retrieved_docs:
            with st.spinner("🤖 Generating answer with LLM..."):
                response = generate_answer(
                    user_input,
                    retrieved_docs,
                    st.session_state.messages
                )

            # Show retrieval info
            with st.expander("🔍 Retrieved Sources"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Source {i+1}** (Score: {doc['score']:.2f}): {doc['question']}")
        else:
            response = "❓ I couldn't find relevant information. Try rephrasing or ask about ML, Deep Learning, NLP, or Data Science!\n\n💡 Check the roadmaps in the sidebar! 🗺️"

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Memory Summary every 5 questions
    if st.session_state.question_count % 5 == 0:
        user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        with st.chat_message("assistant"):
            summary = f"📊 **Session Summary** (after {st.session_state.question_count} questions):\n\nTopics covered: {', '.join(user_msgs[-5:])}\n\n💡 Would you like a learning roadmap? Check the sidebar! 🗺️"
            st.markdown(summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})