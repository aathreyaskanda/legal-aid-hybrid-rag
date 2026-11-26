# Legal Aid Hybrid RAG (India Focus)

A Hybrid Retrieval-Augmented Generation (RAG) legal assistant tailored to **Indian law**. Built with **Streamlit**, **Neo4j**, **SentenceTransformers**, **SerpAPI / DuckDuckGo**, and **Groq**, the application combines semantic graph retrieval (with neighbor expansion) and live web search to generate structured, citation-backed legal guidance—and can dynamically grow its Neo4j knowledge graph from newly extracted legal facts.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture & Workflow](#architecture--workflow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Disclaimer](#disclaimer)

---

## Key Features

- **Hybrid Retrieval (Graph + Web):** Queries a Neo4j native cosine vector index (`all-MiniLM-L6-v2`, 384 dimensions) for the top 5 relevant `Entity` nodes, expands each hit with up to 3 connected 1-hop graph neighbors, and augments the context with live web search results.
- **Resilient Web Search:** Uses **SerpAPI** (Google Search) as the primary web retriever and automatically falls back to **DuckDuckGo** (`in-en` region) if SerpAPI is unavailable or unconfigured.
- **Structured Indian Legal Responses:** Uses Groq (`llama-3.1-8b-instant`) to synthesize evidence strictly around Indian statutes (Constitution of India, IPC, CrPC, Evidence Act, and judicial principles) into 7 standardized sections with inline `[GRAPH#]` and `[WEB#]` citations:
  1. Summary
  2. Relevant Laws & Legal Basis (India)
  3. Rights, Duties & Procedures
  4. Step-by-Step Practical Guidance
  5. Required Documents
  6. Special Notes / Warnings
  7. Citations
- **Self-Updating Knowledge Graph:** After answering a query, users can optionally extract `(subject, predicate, object, confidence)` triples from web results, validate and sanitize predicates, ingest an adaptive number of triples into Neo4j, and automatically compute and store embeddings for the new nodes.
- **Interactive UI & Diagnostics:** Streamlit interface featuring live sidebar status checks (Neo4j connectivity, vector index presence, Groq API key, web search readiness), expandable source viewers, and session query history.

---

## Architecture & Workflow

```text
User Query (Streamlit UI)
        │
        ├──► 1. Graph Retrieval (Neo4j Vector Index)
        │       • Embed query with all-MiniLM-L6-v2 (384-dim)
        │       • Top-5 cosine similarity nodes + up to 3 neighbors each
        │
        ├──► 2. Live Web Search
        │       • SerpAPI (Google Search, top 8) ──► DuckDuckGo fallback (in-en)
        │
        ▼
3. Context Assembly & Answer Generation (Groq: llama-3.1-8b-instant)
        • Produces 7-section Indian legal guide with [GRAPH#] & [WEB#] citations
        │
        ▼
4. Optional Graph Enrichment (User Confirmed)
        • Extract Indian-law triples from web snippets via Groq (JSON)
        • Validate & sanitize predicates ──► MERGE into Neo4j
        • Recompute & store vector embeddings for newly added nodes
