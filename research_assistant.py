# research_assistant.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

import os

class ResearchAssistant:
    def __init__(self, document_path, chunk_size=1000, chunk_overlap=200):
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None

    # Step 1: Load document
    def load_document(self):
        if not os.path.exists(self.document_path):
            raise FileNotFoundError(f"{self.document_path} not found.")

        with open(self.document_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    # Step 2: Split into chunks
    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)
        return chunks

    # Step 3: Create embeddings + FAISS index
    def build_vectorstore(self, chunks):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")  
        self.vectorstore = FAISS.from_texts(chunks, embeddings)
        print("FAISS index created successfully.")

    # Save FAISS index
    def save_index(self, folder="faiss_index"):
        if self.vectorstore is None:
            raise ValueError("Vectorstore has not been created yet.")
        self.vectorstore.save_local(folder)
        print(f"FAISS index saved to folder: {folder}")

    # Load FAISS index
    def load_index(self, folder="faiss_index"):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
    
    # Get retriever
    def get_retriever(self, k=3):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not loaded. Call load_index() first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    # Get LLM
    def get_llm(self):
        llm = ChatOllama(model="llama3")
        return llm
    
    # Answer question
    def answer_question(self, question, k=3):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not loaded. Call load_index() first.")

        retriever = self.get_retriever(k)
        llm = self.get_llm()

        # Retrieve relevant passages
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"Use the following context to answer concisely: {context} \n Question: {question}"

        response = llm.invoke(prompt)

        return response.content