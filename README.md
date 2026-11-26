# Legal Aid Hybrid RAG (India Focus)

A Hybrid Retrieval-Augmented Generation (RAG) legal assistant focused on **Indian law**. Built with **Streamlit**, **Neo4j**, **SentenceTransformers**, **SerpAPI / DuckDuckGo**, and **Groq**, the application combines semantic knowledge-graph retrieval (with 1-hop neighbor expansion) and live web search to produce structured, citation-backed legal guidance—and lets users dynamically expand the Neo4j knowledge graph with newly extracted legal triples.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Disclaimer](#disclaimer)

---

## Overview

Navigating legal procedures, statutory rights, and required documentation in India can be difficult when information is scattered across bare acts, court rulings, and government portals. **Legal Aid Hybrid RAG** combines three components:

1. **Structured Domain Memory (Neo4j):** Cosine vector search over indexed `Entity` nodes combined with graph traversal to pull related legal context.
2. **Real-Time Web Evidence (SerpAPI and DuckDuckGo):** Live search results to cover recent updates, schemes, or topics not yet in the graph.
3. **Structured Synthesis (Groq LLM):** Generates a 7-section legal response grounded in Indian law with inline `[GRAPH#]` and `[WEB#]` citations.

---

## Key Features

- **Hybrid Graph and Web Retrieval:** Embeds user queries with `all-MiniLM-L6-v2` (384 dimensions), retrieves the top 5 matching `Entity` nodes from Neo4j, expands each node with up to 3 connected neighbors, and merges them with up to 8 live web results.
- **Automatic Web Search Fallback:** Uses **SerpAPI** (Google Search) as the primary search provider and automatically falls back to **DuckDuckGo** (`in-en` region, moderate safesearch) if SerpAPI is unavailable or not configured.
- **Standardized 7-Section Legal Answers:** Powered by Groq (`llama-3.1-8b-instant`), every response follows a consistent structure tailored to Indian law (Constitution of India, IPC, CrPC, Evidence Act, and judicial principles):
  1. Summary
  2. Relevant Laws & Legal Basis (India)
  3. Rights, Duties & Procedures
  4. Step-by-Step Practical Guidance
  5. Required Documents
  6. Special Notes / Warnings
  7. Citations (`[GRAPH#]` and `[WEB#]`)
- **Self-Updating Knowledge Graph:** Users can click **Yes, ingest** to extract `(subject, predicate, object, confidence)` triples from web snippets via Groq, validate and normalize predicates (`UPPER_SNAKE_CASE`), merge an adaptive count of triples into Neo4j, and immediately compute and store 384-dim embeddings on the new nodes.
- **Live System Diagnostics and History:** The Streamlit sidebar checks Neo4j connectivity, vector index presence, Groq API key status, and web search readiness in real time, and lets users revisit previous session queries and answers.

---

## Architecture

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
        • Validates & sanitizes predicates and confidence [0, 1]
        • MERGEs Entity nodes and relationships into Neo4j
        • Encodes and writes vector embeddings for all newly added nodes
```

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit (`1.49.1`) |
| **Graph and Vector Store** | Neo4j (`neo4j` driver `6.0.3`, native cosine vector index) |
| **Embedding Model** | SentenceTransformers (`all-MiniLM-L6-v2`, 384-dim) |
| **LLM Inference** | Groq API (`llama-3.1-8b-instant`) |
| **Web Search** | SerpAPI (primary) + `duckduckgo_search` (`8.1.1`, fallback) |
| **Environment Management** | `python-dotenv` (`1.1.1`) |

---

## Project Structure

```text
.
├── finalapp.py          # Main Streamlit Hybrid RAG application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template (safe to commit)
├── .gitignore           # Excludes .env, virtual environments, caches, and logs
└── README.md            # Project documentation
```

---

## Installation

### Prerequisites
- **Python 3.10+**
- **Neo4j 5.11+** or **Neo4j AuraDB** (with native vector index support)
- **Groq API Key** ([console.groq.com](https://console.groq.com))
- **SerpAPI Key** *(optional — falls back to DuckDuckGo if omitted)*

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment:**

   Ubuntu / Linux / macOS:
   ```bash
   python3 -m venv rag-env
   source rag-env/bin/activate
   ```

   Windows (PowerShell):
   ```powershell
   python -m venv rag-env
   .\rag-env\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `GROQ_API_KEY` | Yes | — | Groq API key for LLM generation and triple extraction |
| `SERP_API_KEY` | No | — | SerpAPI key for Google Search (falls back to DuckDuckGo if blank) |
| `NEO4J_URI` | Yes | — | Neo4j connection URI (e.g., `neo4j+s://<id>.databases.neo4j.io`) |
| `NEO4J_USER` | Yes | — | Neo4j username (typically `neo4j`) |
| `NEO4J_PASSWORD` | Yes | — | Neo4j database password |
| `NEO4J_DATABASE` | No | `neo4j` | Target Neo4j database name |
| `NEO4J_DOC_LABEL` | No | `Entity` | Node label indexed for vector search and triple ingestion |
| `NEO4J_VECTOR_INDEX_NAME` | No | `entity_embeddings` | Name of the Neo4j native vector index |
| `EMBED_MODEL_NAME` | No | `all-MiniLM-L6-v2` | SentenceTransformer model used for embeddings |
| `EMBEDDING_DIM` | No | `384` | Vector dimension matching `EMBED_MODEL_NAME` |

> **Security Note:** Never commit `.env` to version control. It is excluded by `.gitignore`.

---

## Usage

Start the Streamlit app:

```bash
streamlit run finalapp.py
```

1. **Check System Status:** Confirm **Neo4j**, **Vector Index**, **Groq API**, and **Web Search** show green indicators in the left sidebar.
2. **Ask a Question:** Enter an Indian legal question and click **Get Answer**.
3. **Review Sources:** Expand **Graph Results** (including 1-hop neighbors) and **Web Results** to inspect the retrieved evidence.
4. **Update the Graph (Optional):** Click **Yes, ingest** to extract and store new Indian-law triples and embeddings in Neo4j, or **No** to skip.
5. **View History:** Select earlier questions from the **Query History** dropdown in the sidebar.

---

## Disclaimer

This project is intended for **educational and informational purposes only** and does not constitute legal advice. For legal representation or official guidance, consult a qualified advocate or contact your nearest Legal Services Authority (NALSA / SLSA / DLSA).
