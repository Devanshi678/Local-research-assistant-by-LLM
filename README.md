# Local-research-assistant-by-LLM
Offline research assistant using LLaMA 3, FAISS, and Retrieval-Augmented Generation (RAG) to answer questions from a local textbook dataset.
## 📌 Project Overview
This project implements a **local research assistant** that can answer questions about a Software Engineering textbook chapter using a **Retrieval-Augmented Generation (RAG)** pipeline.  
The system runs **entirely offline on a single machine**, powered by **LLaMA 3 via Ollama**, without relying on any cloud-based APIs.

Instead of sending data to external services, the assistant retrieves relevant information from a **locally indexed textbook** and generates grounded, context-aware answers. This project demonstrates how modern Transformer-based LLMs can be combined with retrieval techniques to support **long-document reasoning**, **follow-up questions**, and **summarization** in a practical setting.

---

## 🎯 Key Features
- Fully **local and offline** LLM-based research assistant
- Retrieval-Augmented Generation (RAG) for accurate, grounded answers
- Supports:
  - Factual question answering
  - Follow-up questions using prior context
  - Summarization of long content
- Handles documents **larger than the model’s context window**
- Clean, modular Python codebase

---

## 🧠 Machine Learning & NLP Techniques Used
- **Transformer-based Large Language Model**
  - LLaMA 3 running locally via Ollama
- **Retrieval-Augmented Generation (RAG)**
  - Separates retrieval from generation for improved accuracy
- **Text Embeddings**
  - Generated using `nomic-embed-text`
- **Vector Similarity Search**
  - FAISS for fast nearest-neighbor retrieval
- **Semantic Text Chunking**
  - RecursiveCharacterTextSplitter for meaningful document splits
- **Attention-Based Reasoning**
  - Enables long-range dependency handling beyond RNN/LSTM limitations

---

## 🛠️ Tools & Technologies
- **Python**
- **Ollama** (Local LLM runtime)
- **LLaMA 3**
- **LangChain**
- **FAISS (CPU)**
- **nomic-embed-text**
- **VS Code**

---

## 📂 Dataset
- Custom-cleaned Software Engineering textbook chapter (~8,000 words)
- Covers:
  - Software processes
  - Development life cycles
  - Waterfall, Incremental, and Agile models
  - Requirements engineering, design, testing, and maintenance

The text was cleaned and preprocessed to ensure high-quality embeddings and retrieval performance.
Because the system is built using a Retrieval-Augmented Generation (RAG) pipeline, it does not rely on pre-trained knowledge alone. Instead, it answers questions only from the indexed document, making it domain-specific and grounded.

---

## 📁 Project Structure
Project3/
├── research_assistant.py # Core RAG + LLM pipeline
├── run_research_assistant.py # Example queries and demo execution
├── setup_index.py # Dataset processing and FAISS index creation
├── Software_Processes_Clean.txt # Cleaned textbook dataset
├── faiss_index/ # Saved FAISS vector index
├── Project3_Report.pdf # Full project documentation


---

## 🚀 How to Run (Single Machine Setup)

### 1️⃣ Install Dependencies
Run inside a virtual environment (recommended):
```bash
pip install langchain-text-splitters
pip install langchain-ollama
pip install langchain-community
pip install faiss-cpu
pip install ollama

### 2️⃣ Start Ollama Server
In a separate terminal:
ollama server

Pull required models:
ollama pull nomic-embed-text
ollama pull llama3

### 3️⃣ Build the FAISS Index (One-Time Step)
python setup_index.py

Expected output:
Document loaded.
Total chunks created: <number>
FAISS index created successfully.
FAISS index saved to folder: faiss_index

This step only needs to be repeated if the dataset changes.

---

### 4️⃣ Run the Research Assistant
python run_research_assistant.py

The script demonstrates:
A factual question
A follow-up question using earlier context
A summarization request

## 📖 What I Learned from This Project 
- Transformers outperform RNNs/LSTMs in handling long-range dependencies due to self-attention.
- Even powerful LLMs benefit significantly from retrieval, especially for domain-specific or long documents.
- RAG systems reduce hallucinations by grounding answers in real text.
- Running LLMs locally is practical and effective for privacy-preserving AI systems.
- FAISS indexing and chunking strategies have a major impact on answer quality.
- This project deepened my understanding of how attention + retrieval fundamentally changes how models access and reason over information.

---
