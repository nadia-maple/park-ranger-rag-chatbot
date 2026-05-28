# 🌲 PARK RANGER AI ASSISTANT - Hybrid RAG Chatbot

## 📖 Project Description
Park Ranger AI is an interactive, educational chatbot designed to answer questions about National Parks in the United States and Canada. Built using a **Retrieval-Augmented Generation (RAG)** pipeline, it acts as a virtual park ranger by grounding its answers in real documentation.

The system uses a **Hybrid RAG approach**:
1. It first searches a structured dataset of QA pairs for high-confidence matches.
2. If no direct match is found, it falls back to semantic vector search across chunked park documentation using **ChromaDB**.

The UI is built with **Streamlit** and the intelligence is powered by **Google's Gemini 2.5 Flash** model.

### ✨ Key Features
* **Hybrid Retrieval:** Combines exact-match QA lookups with semantic vector search.
* **Source Transparency:** Explicitly cites the documents, pages, and URLs used to generate each answer.
* **Conversation Memory:** Remembers the last few turns of the conversation for follow-up questions.
* **Local Vector Database:** Uses `sentence-transformers/all-MiniLM-L6-v2` and ChromaDB for fast, lightweight, and free local text embedding.

📂 Project Structure

park-ranger-rag-chatbot/
├── app.py                      # Main Streamlit user interface
├── requirements.txt            # Python dependencies
├── README.md
├── src/
│   ├── vector_store.py         # Script to chunk text, create embeddings, and build Chroma DB
│   └── chatbot.py              # Core LLM logic, Chroma querying, and prompt engineering
├── data/
│   ├── qa-combined-top-parks.csv   # Structured QA pairs for the hybrid fallback
│   └── processed/
│       ├── canada-top-parks.json
│       └── us-top-parks.json
├── chroma_parks_db/            # Generated local vector database (created by vector_store.py)
├── notebooks/
│   └── scraper_and_app.ipynb
└── chunks_embeddings_outputs/


## 🚀 How to Run Locally

### 1. Prerequisites
* **Python 3.11** is recommended for compatibility with all underlying machine learning libraries (specifically PyTorch and ONNX).
* A **Google Gemini API Key**. You can get one at Google AI Studio(https://aistudio.google.com/app/apikey).

### 2. Create and Activate the Virtual Environment
Open your terminal/command prompt in the project folder and run:

**Windows:**
```bash
py -3.11 -m venv venv
venv\Scripts\activate

(Note: If you are on Mac/Linux, use python3.11 -m venv venv and source venv/bin/activate)

3. Install Dependencies
First, upgrade your core Python tooling:

```bash
python -m pip install --upgrade pip setuptools wheel

You can install the required packages using the requirements.txt file:

```bash
pip install -r requirements.txt

Alternatively, if you are building the environment from scratch, install these exact package versions to ensure compatibility:

```bash
pip install streamlit==1.57.0
pip install sentence-transformers==3.0.1
pip install chromadb==0.5.5
pip install torch==2.3.1
pip install transformers==4.44.2
pip install pyarrow==17.0.0
pip install numpy==1.26.4
pip install scikit-learn==1.5.1
pip install pandas
pip install google-genai


(Windows Users: If you encounter a c10.dll error with PyTorch, install the CPU-only version by running: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu)

4. Build the Vector Database (First Time Only)
Before running the app for the first time, you need to chunk the park data and build the local ChromaDB vector store:

```bash
python src/vector_store.py

This will create a chroma_parks_db/ folder in your directory containing the vectorized knowledge base.

5. Launch the App
Start the Streamlit server:

```bash
streamlit run app.py

The app will automatically open in your default web browser at http://localhost:8501. Enter your Gemini API key in the sidebar to start chatting!



---
⚠️ Disclaimer
This is an unofficial, educational project. The information provided is sourced from Parks Canada and the US National Park Service but is not endorsed by or affiliated with either government. Always verify critical travel, safety, and permit information on official government websites.
