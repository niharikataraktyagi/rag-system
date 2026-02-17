# RAG System with Local Embeddings

A lightweight Retrieval-Augmented Generation (RAG) system built using local embeddings.

## Features
- Local embedding generation
- Document ingestion pipeline
- Vector similarity search
- Query interface
- Clean modular structure

## Project Structure

rag-system/
│
├── app/
│   ├── ingest.py
│   └── query.py
├── data/
│   └── sample.txt
├── .gitignore
└── README.md

## How It Works

1. Ingest documents
2. Convert text to embeddings
3. Store vectors
4. Retrieve relevant context for queries

## Run the Project

Create virtual environment:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Run ingestion:
python app/query.py


