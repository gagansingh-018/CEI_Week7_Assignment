# RAG Document Question Answering System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system
that answers questions from custom PDF documents.

The system retrieves relevant document chunks using semantic search
and generates grounded answers using a language model.

---

## Features

- PDF document ingestion
- Text chunking
- Semantic embeddings
- FAISS vector database
- Similarity search
- Context-aware answer generation
- Streamlit user interface

---

## Technologies Used

- Python
- LangChain
- Sentence Transformers
- FAISS
- HuggingFace Transformers
- Streamlit

---

## Workflow

1. Load PDF
2. Split text into chunks
3. Generate embeddings
4. Store embeddings in FAISS
5. Accept user question
6. Retrieve relevant chunks
7. Generate grounded answer

---

## Run the Project

```bash
pip install -r requirements.txt
