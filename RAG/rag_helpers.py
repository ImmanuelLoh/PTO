# rag_eval.py
import os, re, json, time
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from UnifiedCFOAgent import query_cfo_agent

# Global OpenAI client
client = OpenAI()

# ============================
#   EMBEDDINGS
# ============================
def embed_texts(texts):
    """Return list of embeddings for given texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]

# ============================
#   RAG TRIAD (COSINE)
# ============================
def compute_context_relevance(query, retrieved_texts):
    """How relevant retrieved docs are to query using cosine similarity."""
    texts = [query] + retrieved_texts
    embeddings = embed_texts(texts)
    query_vec, doc_vecs = embeddings[0], embeddings[1:]
    sims = cosine_similarity([query_vec], doc_vecs)[0]
    return float(np.mean(sims))

def compute_answer_relevance(query, answer):
    """How relevant final answer is to query."""
    query_vec, answer_vec = embed_texts([query, answer])
    return float(cosine_similarity([query_vec], [answer_vec])[0][0])

def compute_faithfulness(answer, retrieved_texts):
    """How much the answer is supported by retrieval content."""
    all_text = " ".join(retrieved_texts)
    nums_ans = re.findall(r"\d+\.?\d*", answer)
    nums_ctx = re.findall(r"\d+\.?\d*", all_text)
    if not nums_ans:
        return 1.0
    match_count = sum(1 for n in nums_ans if n in nums_ctx)
    return match_count / len(nums_ans)

# ============================
#   LLM FLOAT SCORERS
# ============================
def llm_float_score(prompt: str):
    """Ask the LLM to return a float from 0–1 only."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Return ONLY a float from 0 to 1."},
            {"role": "user", "content": prompt}
        ],
    )
    raw = response.choices[0].message.content.strip()
    score = float(raw)
    return max(0, min(score, 1))  # clamp to [0,1]

def context_relevance_score(question, context):
    prompt = f"Rate how relevant the CONTEXT is to the QUESTION from 0 to 1.\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nScore:"
    return llm_float_score(prompt)

def answer_relevance_score(original_question, answer):
    prompt = f"Rate how relevant the ANSWER is to the QUESTION from 0 to 1.\n\nQUESTION:\n{original_question}\n\nANSWER:\n{answer}\n\nScore:"
    return llm_float_score(prompt)

def faithfulness_score(context, answer):
    prompt = f"Rate how faithful the ANSWER is to the CONTEXT from 0 to 1.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore:"
    return llm_float_score(prompt)

# ============================
#   LOG HANDLING
# ============================
LOG_FOLDER = "00-data/logs/sections"
LOG_PREFIX = "test_"
LOG_SUFFIX = ".json"

def list_log_numbers():
    """Return all existing log numbers as integers."""
    if not os.path.exists(LOG_FOLDER):
        return []
    logs = []
    for f in os.listdir(LOG_FOLDER):
        if f.startswith(LOG_PREFIX) and f.endswith(LOG_SUFFIX):
            num = int(f[len(LOG_PREFIX):-len(LOG_SUFFIX)])
            logs.append(num)
    return sorted(logs)

def load_log(n):
    path = os.path.join(LOG_FOLDER, f"{LOG_PREFIX}{n}{LOG_SUFFIX}")
    with open(path, "r") as f:
        return json.load(f)

def extract_context_from_logs(log_nums):
    all_context = []
    for n in log_nums:
        log = load_log(n)
        for section in log.get("results", []):
            for rec in section.get("ranking", []):
                txt = rec.get("text", "")
                if txt:
                    all_context.append(txt)
    return "\n".join(all_context)

# ============================
#   CFO AGENT EVALUATION
# ============================
def run_query_and_collect_logs(question, **agent_kwargs):
    # 1. Capture existing logs
    before = set(list_log_numbers())

    # 2. Run the CFO agent
    response = query_cfo_agent(question, **agent_kwargs)

    # 3. Capture logs after
    after = set(list_log_numbers())

    # 4. Newly generated logs for THIS query:
    new_logs = sorted(list(after - before))

    return response, new_logs

def evaluate_rag_for_query(original_question, **agent_kwargs):
    # Run CFO agent
    answer_obj, logs = run_query_and_collect_logs(original_question, **agent_kwargs)

    answer = answer_obj["answer"]
    expanded_query = answer_obj["metadata"].get("expanded_query", original_question)

    context = extract_context_from_logs(logs)

    # RAG TRIAD
    ctx_rel = context_relevance_score(expanded_query, context)
    faith = faithfulness_score(context, answer)
    ans_rel = answer_relevance_score(original_question, answer)

    return {
        "original_query": original_question,
        "expanded_query": expanded_query,
        "logs": logs,
        "answer": answer,
        "context_relevance": ctx_rel,
        "faithfulness": faith,
        "answer_relevance": ans_rel
    }
    
# ============================
#   SAVE RAG RUN LOGS
# ============================

LOG_DIR = "00-data/logs/rag"
os.makedirs(LOG_DIR, exist_ok=True)

def next_log_index():
    """Return next available log index (integer)."""
    os.makedirs(LOG_DIR, exist_ok=True)

    files = [f for f in os.listdir(LOG_DIR) if f.startswith("rag_")]
    if not files:
        return 1

    indices = []
    for f in files:
        # expecting format rag_001_2025-...
        parts = f.split("_")
        try:
            idx = int(parts[1])
            indices.append(idx)
        except:
            pass

    return max(indices) + 1


def save_run_logs(results, answers):
    """Save single optimized RAG log with index + timestamp."""
    os.makedirs(LOG_DIR, exist_ok=True)

    idx = next_log_index()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"rag_{idx}_{timestamp}.json"
    out_path = os.path.join(LOG_DIR, file_name)

    # Build merged log object
    merged = {
        "run_index": idx,
        "timestamp": timestamp,
        "num_queries": len(results),
        "total_latency_s": sum(r["Latency (s)"] for r in results),
        "queries": []
    }

    for r, a in zip(results, answers):
        entry = {
            "Query ID": r["Query ID"],
            "Query Name": r["Query Name"],
            "Scores": {
                "Latency (s)": r["Latency (s)"],
                "Context Relevance": {
                    "cosine": r["Context Relevance (cosine)"],
                    "llm": r["Context Relevance (LLM)"],
                    "merged": r["Context Relevance (Merged)"]
                },
                "Answer Relevance": {
                    "cosine": r["Answer Relevance (cosine)"],
                    "llm": r["Answer Relevance (LLM)"],
                    "merged": r["Answer Relevance (Merged)"]
                },
                "Faithfulness": r["Faithfulness"],
                "Avg Triad (Merged)": r["Avg Triad (Merged)"]
            },
            "Answer": {
                "text": a["Full Answer"],
                "Sources": a["Sources"]
            },
            "Logs Used": r["Log Files"]
        }
        merged["queries"].append(entry)

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Saved combined log:\n - {out_path}")



