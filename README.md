# Mini RAG Engine (Tokenization + Embeddings + Vector Search)

A small project that demonstrates the core building blocks of Retrieval Augmented Generation (RAG).

## What this project does

- Tokenizes text using `tiktoken`
- Shows token IDs and how LLMs read text
- Creates embeddings using `all-MiniLM-L6-v2`
- Stores embeddings for document chunks
- Performs vector similarity search using cosine similarity
- Retrieves the most relevant chunk for a user query

This is the foundation of how modern AI assistants work.

## Features

- Token analyzer (inspect tokens + decoded tokens)
- Embedding generator (convert text to vector)
- Vector search engine (find best matching chunk)
- Simple RAG flow (load text file → chunk → embed → search → return best chunk)

## Tech Stack

- Python
- tiktoken
- sentence-transformers
- sklearn cosine similarity

## How it works (short)

1. Load a text file (dataset)
2. Split it into small chunks
3. Generate embeddings for each chunk
4. Embed the user query
5. Compare query embedding with chunk embeddings
6. Return the most similar chunk

## Why this project matters

This is the exact core system behind:
- RAG chatbots
- AI search tools
- Document Q&A assistants
- Knowledge base AI agents

Understanding this = understanding real AI workflow.

