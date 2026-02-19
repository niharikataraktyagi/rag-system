RAG-Based Document Question Answering System

Overview

This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI, ChromaDB, HuggingFace embeddings, and Llama3 via Ollama. The system allows users to upload documents (TXT or PDF) and ask grounded questions based only on the uploaded content.

Architecture

Document Ingestion

TXT and PDF files supported

Documents chunked into 500-character segments

Embedded using all-MiniLM-L6-v2

Stored in ChromaDB with persistence

Query Flow

Harmful query guardrail (pre-check)

Similarity search (top-k retrieval)

Relevance threshold check

Context-grounded LLM response

Response safety filtering

Guardrails

Blocks harmful queries (e.g., bomb, attack)

Rejects unrelated queries

Logs blocked attempts in guardrails.log
