from research_assistant import ResearchAssistant

assistant = ResearchAssistant("Software_Processes_Clean.txt")

# Step 1: Load
text = assistant.load_document()
print("Document loaded.")

# Step 2: Split
chunks = assistant.split_text(text)
print(f"Total chunks created: {len(chunks)}")

# Step 3: Build FAISS index
assistant.build_vectorstore(chunks)

# Step 4: Save FAISS index
assistant.save_index()