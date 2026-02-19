from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_ollama import OllamaLLM
import os
import logging


# ----------------------------
# Logging Setup (Guardrails)
# ----------------------------
logging.basicConfig(
    filename="guardrails.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


app = FastAPI()

PERSIST_DIRECTORY = "chroma_db"

# ----------------------------
# Embedding Model
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# Load Existing DB If Present
# ----------------------------
if os.path.exists(PERSIST_DIRECTORY):
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
else:
    db = None

# ----------------------------
# LLM (Ollama - Llama3)
# ----------------------------
llm = OllamaLLM(model="llama3")


class QueryRequest(BaseModel):
    question: str


# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Document Ingestion (TXT + PDF)
# ----------------------------
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    global db

    contents = await file.read()
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(contents)

    # Choose loader based on file type
    if file.filename.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        return {"error": "Only .txt and .pdf files are supported"}

    documents = loader.load()

    # Simple chunking
    texts = []
    chunk_size = 500

    for doc in documents:
        content = doc.page_content
        for i in range(0, len(content), chunk_size):
            texts.append(content[i:i + chunk_size])

    db = Chroma.from_texts(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return {"message": "Document ingested successfully"}


# ----------------------------
# Query Endpoint (With Guardrails)
# ----------------------------
@app.post("/query")
def query(request: QueryRequest):
    global db

    if db is None:
        return {"error": "No documents ingested yet"}

    user_question = request.question.lower()

    # ----------------------------
    # Guardrail 0: Block Harmful Questions Immediately
    # ----------------------------
    banned_words = ["kill", "hate", "bomb", "attack", "weapon"]

    for word in banned_words:
        if word in user_question:
            logging.info(f"Blocked harmful query: {request.question}")
            return {
                "error": "I cannot assist with that type of request."
            }

    # ----------------------------
    # Continue Normal Retrieval
    # ----------------------------
    results = db.similarity_search_with_score(request.question, k=2)

    top_score = results[0][1]

    # ----------------------------
    # Guardrail 1: Relevance Check
    # ----------------------------
    if top_score > 1.5:
        logging.info(f"Rejected unrelated query: {request.question}")
        return {"error": "Query unrelated to ingested documents"}

    documents = [doc for doc, score in results]

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{request.question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "question": request.question,
        "answer": response,
        "sources": [doc.page_content for doc in documents]
    }


            
