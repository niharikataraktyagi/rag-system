from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1️⃣ Load document
loader = TextLoader("data/sample.txt")
documents = loader.load()

# 2️⃣ Split document into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# 3️⃣ Use FREE local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
db.persist()

print("Ingestion complete using FREE local embeddings!")
