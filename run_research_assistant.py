from research_assistant import ResearchAssistant

assistant = ResearchAssistant("Software_Processes_Clean.txt")

# Load the FAISS index
assistant.load_index()

print("\n--- Query 1: General factual question ---")
answer1 = assistant.answer_question("What is software engineering?")
print(answer1)

print("\n--- Query 2: Follow-up question ---")
answer2 = assistant.answer_question("According to the document, what are the main phases of the software process?")
print(answer2)

print("\n--- Query 3: Summarization request ---")
answer3 = assistant.answer_question("Summarize the section about software process models.")
print(answer3)
