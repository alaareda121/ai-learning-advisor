# 🎓 AI Learning Advisor

## Overview
An intelligent chatbot powered by **RAG (Retrieval-Augmented Generation)** and **Groq LLM** for answering questions about AI, Machine Learning, Deep Learning, NLP, and Data Science.

## Features
- 🔍 **Semantic Search** — Understands questions even if phrased differently
- 🤖 **RAG + Groq LLM** — Retrieves relevant knowledge and generates personalized answers
- 📊 **Evaluation** — Top-3 Retrieval Accuracy: 100%
- 🗺️ **Learning Roadmaps** — ML, Deep Learning, NLP, Data Science
- 💬 **Memory** — Remembers conversation history
- 🎨 **Professional UI** — Built with Streamlit

## Tech Stack
- Python
- Streamlit
- Sentence Transformers
- Groq LLM (llama-3.3-70b-versatile)
- Scikit-learn
- RAG Architecture

## Knowledge Base
- 112 Q&A covering ML, Deep Learning, NLP, Data Science, and more

## How to Run
1. Clone the repo
2. Install requirements: `pip install streamlit sentence-transformers groq python-dotenv scikit-learn`
3. Add your Groq API key in `.env` file: `GROQ_API_KEY=your_key_here`
4. Run: `streamlit run app.py`
