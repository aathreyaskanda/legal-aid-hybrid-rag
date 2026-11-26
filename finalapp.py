"""
Legal Aid Hybrid RAG — India Focus (Merged Complete Version)
- Preserves sidebar UI with status checks and history
- Uses advanced vector retrieval from Neo4j with neighbor expansion
- Web search with SerpAPI primary and DuckDuckGo fallback
- Multi-section detailed LLM answers via Groq
- Triple extraction and validated ingestion with recommended N logic
- Embeddings updated after ingestion
"""

import os
import re
import json
import time
import uuid
import random
import requests
import tldextract
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import streamlit as st

load_dotenv()

# -----------------------
# CONFIG / ENV
# -----------------------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))
VECTOR_INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX_NAME", "entity_embeddings")
ENTITY_LABEL = os.getenv("NEO4J_DOC_LABEL", "Entity")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# -----------------------
# Utility / caching singletons
# -----------------------
@st.cache_resource
def get_neo4j_driver():
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        raise RuntimeError("Missing Neo4j credentials in environment.")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@st.cache_resource
def get_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------
# Neo4j helpers
# -----------------------
def run_cypher(cypher, params=None):
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        return list(session.run(cypher, params or {}))

def create_vector_index_if_missing():
    cypher = f"""
    CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
    FOR (d:`{ENTITY_LABEL}`)
    ON (d.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIM},
            `vector.similarity_function`: 'cosine'
        }}
    }};
    """
    try:
        run_cypher(cypher)
    except Exception:
        pass

# -----------------------
# Vector retrieval
# -----------------------
def vector_query_nodes(query_text, top_k=5):
    model = get_embed_model()
    qemb = model.encode([query_text], convert_to_numpy=True)[0].tolist()
    cypher = """
    CALL db.index.vector.queryNodes($index, $k, $v)
    YIELD node, score
    RETURN node.uid AS uid, node.name AS name, node.text AS text, score
    """
    recs = run_cypher(cypher, {"index": VECTOR_INDEX_NAME, "k": top_k, "v": qemb})
    return [{"uid": r["uid"], "name": r["name"], "text": r["text"], "score": float(r["score"])} for r in recs]

# -----------------------
# Fetch neighbour texts
# -----------------------
def fetch_neighbours(uid, limit=3):
    cypher = """
    MATCH (n:`Entity` {uid:$uid})--(m:`Entity`)
    WHERE m.text IS NOT NULL AND m.uid <> $uid
    RETURN DISTINCT m.uid AS uid, m.name AS name, m.text AS text LIMIT $limit
    """
    recs = run_cypher(cypher, {"uid": uid, "limit": limit})
    return [{"uid": r["uid"], "name": r["name"], "text": r["text"]} for r in recs]

# -----------------------
# Web search
# -----------------------
def serpapi_search(query, num_results=8):
    if not SERP_API_KEY:
        return []
    try:
        res = requests.get(
            "https://serpapi.com/search",
            params={"engine":"google","q":query,"api_key":SERP_API_KEY,"num":num_results,"hl":"en"},
            timeout=10
        ).json()
        items = []
        for r in res.get("organic_results", [])[:num_results]:
            items.append({"title": r.get("title",""), "snippet": r.get("snippet",""), "url": r.get("link","")})
        return items
    except Exception:
        return []

def duckduckgo_search(query, num_results=8):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region='in-en', safesearch='Moderate', max_results=num_results)
            items = [{"title": r.get("title",""), "snippet": r.get("body",""), "url": r.get("href","")} for r in results]
            return items
    except Exception:
        return []

def search_with_fallback(query, num_results=8):
    r = serpapi_search(query, num_results)
    if r:
        return r
    return duckduckgo_search(query, num_results)

# -----------------------
# LLM: structured detailed answer
# -----------------------
def generate_answer(question, context, max_tokens=900):
    if not GROQ_API_KEY:
        return ("(LLM key not configured) Summary:\n\n"
                "Use the graph and web evidence shown below to answer the user's query.\n\n"
                "Evidence:\n" + context[:1000])
    prompt = f"""
You are an Indian legal expert.

Use ONLY Indian laws, Constitution, IPC, CrPC, Evidence Act, judicial principles.
Ignore foreign non-Indian statutes. Always cite evidence like [GRAPH#1], [WEB#2].

Produce answer with EXACT sections:

1. 📝 SUMMARY
2. 📚 RELEVANT LAWS & LEGAL BASIS (INDIA)
3. 🏛 RIGHTS, DUTIES & PROCEDURES
4. ⚖ STEP-BY-STEP PRACTICAL GUIDANCE
5. 🪪 REQUIRED DOCUMENTS
6. 🚨 SPECIAL NOTES / WARNINGS
7. 🔗 CITATIONS (list GRAPH# and WEB# used)

EVIDENCE:
{context}

QUESTION:
{question}
"""
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {"model":"llama-3.1-8b-instant",
                   "messages":[{"role":"user","content":prompt}],
                   "max_tokens":max_tokens, "temperature":0.2}
        r = requests.post(GROQ_CHAT_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"(LLM call error) {e}\n\nPartial context:\n{context[:1200]}"

# -----------------------
# Triple extraction
# -----------------------
TRIPLE_PROMPT = """
You are an expert at extracting concise factual triples from legal text.
Return ONLY a JSON array of objects: [{subject, predicate, object, confidence}]

Rules:
- Only include facts about INDIAN law/courts/procedures/rights.
- predicate must be a single token (letters/numbers, may have one underscore).
- subject and object must be short human-readable strings.
- confidence: float between 0 and 1.
- Output strict JSON array ONLY.

TEXT:
\"\"\"{text}\"\"\"
"""

def extract_triples_from_text(text):
    if not GROQ_API_KEY:
        return []
    prompt = TRIPLE_PROMPT.format(text=(text[:8000] if isinstance(text,str) else ""))
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {"model":"llama-3.1-8b-instant",
                   "messages":[{"role":"user","content":prompt}],
                   "max_tokens":1024,"temperature":0.0}
        r = requests.post(GROQ_CHAT_URL, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        content = r.json().get("choices",[{}])[0].get("message",{}).get("content","")
        start = content.find("[")
        if start>=0:
            content = content[start:]
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except Exception:
        return []

# -----------------------
# Triple validation
# -----------------------
PRED_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
def sanitize_pred(pred):
    if not pred:
        return None
    p = str(pred).strip().replace(" ", "_")
    p = re.sub(r"[^A-Za-z0-9_]", "", p)
    return p.upper() if PRED_RE.match(p) else None

def validate_triple(t):
    try:
        s = str(t.get("subject","")).strip()
        o = str(t.get("object","")).strip()
        p = sanitize_pred(t.get("predicate",""))
        conf = float(t.get("confidence", 1.0))
    except Exception:
        return None
    if not (s and o and p):
        return None
    conf = max(0.0, min(1.0, conf))
    return {"subject": s, "predicate": p, "object": o, "confidence": conf}

# -----------------------
# Triple summary
# -----------------------
def triple_to_summary(tr):
    s = f"{tr['subject']} {tr['predicate'].lower().replace('_',' ')} {tr['object']}"
    return s if len(s)<=240 else s[:237]+"..."

# -----------------------
# Ingest triples
# -----------------------
def ingest_triples(triples, source_url=None, batch=32):
    if not triples:
        return {"ingested":0,"nodes":[],"new_triples":[]}
    driver = get_neo4j_driver()
    ts = datetime.utcnow().strftime(TS_FMT)
    ingested_nodes = set()
    new_triples_records = []

    for i in range(0, len(triples), batch):
        chunk = triples[i:i+batch]
        with driver.session(database=NEO4J_DATABASE) as session:
            for t in chunk:
                subj = t["subject"]
                obj = t["object"]
                pred = t["predicate"]
                conf = float(t.get("confidence",1.0))
                s_summary = triple_to_summary(t)
                o_summary = s_summary if subj==obj else s_summary

                cypher = (
                    f"MERGE (s:{ENTITY_LABEL} {{name:$sname}}) "
                    f"ON CREATE SET s.text=$stext, s.first_seen=$ts, s.uid = randomUUID() "
                    f"SET s.last_seen=$ts, s.confidence = coalesce(s.confidence, $conf) "
                    f"MERGE (o:{ENTITY_LABEL} {{name:$oname}}) "
                    f"ON CREATE SET o.text=$otext, o.first_seen=$ts, o.uid = randomUUID() "
                    f"SET o.last_seen=$ts, o.confidence = coalesce(o.confidence, $conf) "
                    f"MERGE (s)-[r:{pred}]->(o) "
                    f"SET r.source = $src, r.timestamp=$ts, r.confidence=$conf "
                )
                params = {"sname":subj, "stext":s_summary, "oname":obj, "otext":o_summary,
                          "ts":ts, "conf":conf, "src": source_url or ""}
                try:
                    session.run(cypher, params)
                    ingested_nodes.add(subj); ingested_nodes.add(obj)
                    new_triples_records.append({"s":subj,"p":pred,"o":obj,"conf":conf})
                except Exception:
                    continue

    return {"ingested": len(new_triples_records), "nodes": list(ingested_nodes), "new_triples": new_triples_records}

# -----------------------
# Update embeddings
# -----------------------
def update_embeddings_for_node_names(names, batch=64):
    if not names:
        return []
    model = get_embed_model()
    driver = get_neo4j_driver()
    updated = []
    for i in range(0, len(names), batch):
        chunk = names[i:i+batch]
        with driver.session(database=NEO4J_DATABASE) as session:
            recs = session.run(f"MATCH (n:{ENTITY_LABEL}) WHERE n.name IN $names "
                               "RETURN n.name AS name, coalesce(n.text,n.name) AS text", {"names":chunk})
            rows = list(recs)
            if not rows:
                continue
            texts = [r["text"] for r in rows]
            names2 = [r["name"] for r in rows]
            embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            for nm, emb in zip(names2, embs):
                try:
                    session.run(f"MATCH (n:{ENTITY_LABEL} {{name:$name}}) SET n.embedding=$emb, n.embedding_dim=$dim",
                                {"name":nm, "emb":emb.tolist(), "dim":len(emb)})
                    updated.append(nm)
                except Exception:
                    pass
    return updated

# -----------------------
# Recommend ingest count
# -----------------------
def recommend_ingest_count(extracted_triples, graph_hits):
    ext = len(extracted_triples)
    if ext >= 20:
        return 20
    if 4 <= ext < 20:
        if ext %2 == 1:
            return ext
        return ext-1 if ext-1>=4 else ext
    if not graph_hits:
        return 4 if ext>0 else 0
    avg_score = sum([h.get("score",0) for h in graph_hits])/len(graph_hits)
    if avg_score < 0.60 and ext>0:
        return 4
    return 0

# -----------------------
# Status check functions
# -----------------------
def check_neo4j():
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        return True
    except Exception:
        return False

def check_vector_index():
    try:
        run_cypher(f"SHOW INDEXES WHERE name = '{VECTOR_INDEX_NAME}'")
        return True
    except Exception:
        return False

def check_groq():
    return bool(GROQ_API_KEY)

def check_web():
    return bool(SERP_API_KEY) or True  # DDG always available

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Legal Aid Hybrid RAG", layout="wide")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------
# LEFT SIDEBAR (STATUS + HISTORY)
# ---------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ System Status")

    st.write("Neo4j:", "🟢 Connected" if check_neo4j() else "🔴 Not Connected")
    st.write("Vector Index:", "🟢 Found" if check_vector_index() else "🔴 Missing")
    st.write("Groq API:", "🟢 OK" if check_groq() else "🔴 Missing Key")
    st.write("Web Search:", "🟢 Ready" if check_web() else "🔴 Error")

    st.markdown("---")
    st.markdown("### 📝 Query History")

    if st.session_state.history:
        choice = st.selectbox("Select past query:", ["(none)"] + [h["q"] for h in st.session_state.history])
        if choice != "(none)":
            item = next(h for h in st.session_state.history if h["q"] == choice)
            st.markdown("#### Full Answer")
            st.write(item["a"])

# ---------------------------------------
# MAIN AREA
# ---------------------------------------
st.title("HYBRID RAG LEGAL-AID")

query = st.text_input("Ask your legal question:")

if st.button("Get Answer"):
    with st.spinner("Processing your query..."):
        # GRAPH SEMANTIC SEARCH
        create_vector_index_if_missing()
        graph_hits = vector_query_nodes(query, top_k=5)

        # WEB SEARCH
        web_hits = search_with_fallback(query, num_results=8)

        # Build full context
        ctx = ""
        citation_id = 1
        for g in graph_hits:
            ctx += f"[GRAPH#{citation_id}] UID:{g['uid']} — {g['name']}\n{g['text']}\n\n"
            citation_id += 1
            neighs = fetch_neighbours(g['uid'], limit=3)
            for n in neighs:
                ctx += f"[GRAPH#{citation_id}] (neighbour) UID:{n['uid']} — {n['name']}\n{n['text']}\n\n"
                citation_id += 1

        for w in web_hits:
            ctx += f"[WEB#{citation_id}] {w.get('title','')}\n{w.get('snippet','')}\nURL: {w.get('url','')}\n\n"
            citation_id += 1

        # Generate detailed answer
        ans = generate_answer(query, ctx)

        # Save to history
        st.session_state.history.append({"q": query, "a": ans})

        # Display answer
        st.markdown("## 💬 Answer")
        st.markdown(ans)

        # Show sources
        st.markdown("---")
        st.markdown("### 📖 Sources Used")
        
        st.markdown("#### Graph Results")
        for idx, g in enumerate(graph_hits, start=1):
            with st.expander(f"GRAPH#{idx} — UID: {g.get('uid')} — Score: {g.get('score'):.3f}"):
                st.write(g.get('text') or "")
                neighs = fetch_neighbours(g.get('uid'), limit=3)
                if neighs:
                    st.write("**Neighbours:**")
                    for n in neighs:
                        st.write(f"- `{n['uid']}` {(n['text'] or '')[:200]}")

        st.markdown("#### Web Results")
        for i, w in enumerate(web_hits, start=1):
            with st.expander(f"WEB#{i} — {w.get('title','(no title)')}"):
                st.write(w.get('snippet', ''))
                st.write(f"**URL:** {w.get('url', '')}")

        # INGEST SECTION
        st.markdown("---")
        st.markdown("### 📥 Update Graph with Extracted Triples?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Yes, ingest"):
                with st.spinner("Extracting and ingesting triples..."):
                    extracted_raw = []
                    for w in web_hits:
                        text_blob = (w.get("title", "") + "\n\n" + w.get("snippet", "")).strip()
                        triples = extract_triples_from_text(text_blob)
                        for t in triples:
                            vt = validate_triple(t)
                            if vt:
                                vt["_url"] = w.get("url", "")
                                extracted_raw.append(vt)
                    
                    rec_n = recommend_ingest_count(extracted_raw, graph_hits)
                    to_ingest = extracted_raw[:rec_n]
                    added = ingest_triples(to_ingest, source_url=", ".join([w.get("url","") for w in web_hits]))
                    
                    all_nodes = list(set([t["subject"] for t in to_ingest] + [t["object"] for t in to_ingest]))
                    update_embeddings_for_node_names(all_nodes)

                    st.success(f"✅ Added {added['ingested']} triples to the graph!")
                    
                    if added["new_triples"]:
                        st.markdown("**Top 5 Added Triples:**")
                        for t in added["new_triples"][:5]:
                            st.write(f"- `{t['s']}` — **{t['p']}** → `{t['o']}` (confidence: {t.get('conf',1.0):.2f})")

        with col2:
            if st.button("No"):
                st.info("Skipped updating graph.")

