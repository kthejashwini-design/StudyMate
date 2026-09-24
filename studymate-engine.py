# Author Theju
import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat

# 1. Load the embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# 2. Connect to ChromaDB
db = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = db.get_collection(
    name="my_presentations"
)

print("Connected to ChromaDB!")


# 3. Chat loop
print("\nPPT RAG Chatbot is ready!")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Chatbot closed!")
        break

    if not question:
        continue

    if collection.count() == 0:
        print("Your ChromaDB collection is empty.")
        continue

    # 4. Convert question into embedding
    query_embedding = model.encode(
        question
    ).tolist()

    # 5. Retrieve relevant PPT chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(4, collection.count())
    )

    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    # 6. Prepare context
    context_parts = []

    for doc, info in zip(documents, metadata):

        source = info.get("source", "Unknown")
        page = info.get("page", "Unknown")

        context_parts.append(
            f"Source: {source}, "
            f"Slide/Page: {page}\n{doc}"
        )

    context = "\n\n".join(context_parts)

    # 7. Send context and question to local LLM
    try:

        response = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful study assistant. "
                        "Answer using the provided PPT/PDF "
                        "context. If the answer is missing, "
                        "say you could not find it in the "
                        "presentations. Explain simply and "
                        "mention source filename and "
                        "slide/page when possible."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ]
        )

        print("\nChatbot:")
        print(response.message.content)

    except Exception as error:
        print("\nError:", error)

    print("\n" + "-" * 50 + "\n")