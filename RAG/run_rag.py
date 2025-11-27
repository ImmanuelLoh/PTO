# rag_benchmark.py

import time
import pandas as pd
import numpy as np
from openai import OpenAI
from UnifiedCFOAgent import query_cfo_agent

# Import your helpers
from rag_helpers import (
    run_query_and_collect_logs,
    extract_context_from_logs,
    context_relevance_score,
    faithfulness_score,
    evaluate_rag_for_query,
    answer_relevance_score,
    compute_context_relevance,
    compute_answer_relevance,
    compute_faithfulness,
    list_log_numbers,
    load_log
)

# Your notebook’s custom extractor
def extract_texts_from_log(log):
    chunks = []
    for section in log.get("results", []):
        for item in section.get("ranking", []):
            txt = item.get("text", "").strip()
            if txt:
                chunks.append(txt)
    return chunks


client = OpenAI()

# ============================================================
# 2) EXACT NOTEBOOK FUNCTION: compute manual_ctx
# ============================================================
def compute_manual_ctx(benchmark_queries):
    manual_ctx = {}
    for q in benchmark_queries:
        result = evaluate_rag_for_query(
            q["query"],
            cache_threshold=0.85,
            use_cache=False
        )
        manual_ctx[q["id"]] = result["context_relevance"]
    return manual_ctx


# ============================================================
# 3) EXACT NOTEBOOK BIG LOOP → run_full_rag_benchmark()
# ============================================================
def run_full_rag_benchmark(benchmark_queries, manual_ctx):

    previous_logs = set(list_log_numbers())
    results = []
    answers = []

    for q in benchmark_queries:
        print("\n" + "="*100)
        print(f"📊 {q['id']} — {q['name']}")
        print("="*100)

        user_query = q["query"]
        start_time = time.time()

        # Run CFO agent
        response = query_cfo_agent(
            user_query,
            cache_threshold=0.85,
            use_cache=False
        )

        latency = round(time.time() - start_time, 2)
        answer_text = response["answer"].strip()

        # Detect new logs
        current_logs = set(list_log_numbers())
        new_logs = list(current_logs - previous_logs)
        previous_logs = current_logs.copy()

        # Load logs
        logs = [load_log(n) for n in new_logs]

        # Extract context EXACTLY like notebook
        retrieved_texts = []
        for log in logs:
            retrieved_texts.extend(extract_texts_from_log(log))
        if not retrieved_texts:
            retrieved_texts = [""]

        # ========= COSINE TRIAD =========
        ctx_cos = compute_context_relevance(user_query, retrieved_texts)
        faith = compute_faithfulness(answer_text, retrieved_texts)

        # precomputed manual ctx
        ctx_llm = manual_ctx[q["id"]]
        ctx_merged = (ctx_cos + ctx_llm) / 2

        # ========= Answer relevance =========
        ans_cos = compute_answer_relevance(user_query, answer_text)
        ans_llm = answer_relevance_score(user_query, answer_text)
        ans_merged = (ans_cos + ans_llm) / 2

        # ========= Final merged triad =========
        triad_merged = float(np.mean([ctx_merged, ans_merged, faith]))

        # Store results
        results.append({
            "Query ID": q["id"],
            "Query Name": q["name"],
            "Latency (s)": latency,

            "Context Relevance (cosine)": round(ctx_cos, 3),
            "Context Relevance (LLM)": round(ctx_llm, 3),
            "Context Relevance (Merged)": round(ctx_merged, 3),

            "Answer Relevance (cosine)": round(ans_cos, 3),
            "Answer Relevance (LLM)": round(ans_llm, 3),
            "Answer Relevance (Merged)": round(ans_merged, 3),

            "Faithfulness": round(faith, 3),
            "Avg Triad (Merged)": round(triad_merged, 3),

            "Num Logs": len(new_logs),
            "Log Files": new_logs,
        })

        answers.append({
            "Query ID": q["id"],
            "Query Name": q["name"],
            "Full Answer": answer_text,
            "Sources": response.get("metadata", {})
        })

    return pd.DataFrame(results), pd.DataFrame(answers)
