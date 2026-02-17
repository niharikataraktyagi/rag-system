from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load existing vector database
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Ask a question
query = input("Ask a question: ")

results = db.similarity_search(query, k=2)

print("\nTop relevant chunks:\n")

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(doc.page_content)
    print("-" * 40)
