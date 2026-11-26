# Legal Aid Hybrid RAG (India Focus)

A Hybrid Retrieval-Augmented Generation (RAG) legal assistant focused on **Indian law**. Built with **Streamlit**, **Neo4j**, **SentenceTransformers**, **SerpAPI / DuckDuckGo**, and **Groq**, the application combines semantic knowledge-graph retrieval (with 1-hop neighbor expansion) and live web search to produce structured, citation-backed legal guidance—and lets users dynamically expand the Neo4j knowledge graph with newly extracted legal triples.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Data Flow](#architecture--data-flow)
- [Self-Updating Knowledge Graph Logic](#self-updating-knowledge-graph-logic)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

---

## Overview

Navigating legal procedures, statutory rights, and required documentation in India can be difficult, especially when information is scattered across bare acts, court rulings, and government portals. **Legal Aid Hybrid RAG** addresses this by uniting:

1. **Structured Domain Memory (Neo4j):** Fast cosine vector search over indexed `Entity` nodes combined with graph traversal to pull related legal context.
2. **Real-Time Web Evidence (SerpAPI + DuckDuckGo):** Live search results to cover recent updates, schemes, or topics not yet in the graph.
3. **Structured Synthesis (Groq LLM):** Generates a clear, 7-section legal response grounded strictly in Indian law with inline `[GRAPH#]` and `[WEB#]` citations.

---

## Key Features

- **Hybrid Graph + Web Retrieval:** Embeds user queries with `all-MiniLM-L6-v2` (384 dimensions), retrieves the top 5 matching `Entity` nodes from Neo4j, expands each node with up to 3 connected neighbors, and merges them with up to 8 live web results.
- **Automatic Web Search Fallback:** Uses **SerpAPI** (Google Search) as the primary search provider and seamlessly falls back to **DuckDuckGo** (`in-en` region, moderate safesearch) if SerpAPI is unavailable or not configured.
- **Standardized 7-Section Legal Answers:** Powered by Groq (`llama-3.1-8b-instant`), every response follows a consistent structure tailored to Indian law (Constitution of India, IPC, CrPC, Evidence Act, and judicial principles):
  1. **Summary**
  2. **Relevant Laws & Legal Basis (India)**
  3. **Rights, Duties & Procedures**
  4. **Step-by-Step Practical Guidance**
  5. **Required Documents**
  6. **Special Notes / Warnings**
  7. **Citations** (`[GRAPH#]` and `[WEB#]`)
- **Human-in-the-Loop Graph Enrichment:** Users can approve extracting `(subject, predicate, object, confidence)` triples from web results, validating and merging them into Neo4j, and automatically generating vector embeddings for new nodes.
- **Live System Diagnostics & History:** The Streamlit sidebar monitors Neo4j connectivity, vector index availability, Groq API key status, and web search readiness in real time, alongside a dropdown to revisit previous session queries and answers.

---

## Architecture & Data Flow

```text
User Query (Streamlit UI)
        │
        ├──► 1. Graph Semantic Search (Neo4j Vector Index)
        │       • Auto-creates cosine vector index if missing
        │       • Query embedded via SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
        │       • Top-5 Entity nodes + up to 3 connected 1-hop neighbors each
        │
        ├──► 2. Live Web Search
        │       • Primary: SerpAPI (Google Search, top 8)
        │       • Fallback: DuckDuckGo Search (in-en, top 8)
        │
        ▼
3. Context Assembly & LLM Generation (Groq: llama-3.1-8b-instant)
        • Combines [GRAPH#] and [WEB#] evidence
        • Outputs 7-section Indian legal guide with inline citations
        │
        ▼
4. Optional Knowledge Graph Update (User Confirmed)
        • Extracts Indian-law factual triples from web snippets (strict JSON)
        • Validates & sanitizes predicates (UPPER_SNAKE_CASE) and confidence [0, 1]
        • MERGEs Entity nodes and relationships into Neo4j
        • Encodes and writes vector embeddings for all newly added nodes
