# 📚 Data Mining Unit 1 RAG Assistant

A Retrieval-Augmented Generation (RAG) based question-answering assistant built around Data Mining Unit 1 lecture materials.

The system retrieves relevant information from the provided PowerPoint slides, reranks the retrieved content using a cross-encoder, and generates grounded answers using Llama 3.2 3B through Ollama.

---

## 🚀 Project Overview

This project demonstrates an end-to-end RAG pipeline for answering questions from Data Mining Unit 1 study materials.

Instead of allowing the language model to answer purely from its pretrained knowledge, the system first retrieves relevant information from the course material and uses that information as context for answer generation.

### Pipeline

```text
PowerPoint Files
       ↓
Text Extraction
       ↓
Document Chunking
       ↓
Sentence Embeddings
       ↓
ChromaDB Vector Database
       ↓
Initial Retrieval (Top 10)
       ↓
Cross-Encoder Reranking
       ↓
Top 5 Relevant Chunks
       ↓
Llama 3.2 3B
       ↓
Grounded Answer + Sources

PowerPoint Documents
        ↓
Text Extraction
        ↓
Chunking
        ↓
Embedding Model
        ↓
ChromaDB
        ↓
Top-10 Retrieval
        ↓
Cross-Encoder Reranking
        ↓
Top-5 Context
        ↓
Llama 3.2 3B
        ↓
Grounded Answer
        ↓
Source Slides