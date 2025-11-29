# rag_benchmark.py

import time
import pandas as pd
import json

import numpy as np
from openai import OpenAI

from typing import List, Optional, TypedDict
import sys, os 
# Add parent directory (PTO-Group) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    load_log,
    save_run_logs
)

class LogExtract(TypedDict):
    expanded_query: str
    chunks: List[str]


# Your notebook’s custom extractor
def extract_texts_from_log(log) -> LogExtract: 
    expanded_query = log.get("expanded_query", "")
    chunks = []
    for section in log.get("results", []):
        for item in section.get("ranking", []):
            txt = item.get("text", "").strip()
            if txt:
                chunks.append(txt)
    #return chunks
    return {
        "expanded_query": expanded_query,
        "chunks": chunks 
    }

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
        print (f" - Answer Relevance (cosine): {ans_cos}") 
        print (f" - Answer Relevance (LLM): {ans_llm}") 
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
        
    save_run_logs(results, answers)

    return pd.DataFrame(results), pd.DataFrame(answers)

def run_full_rag_benchmark_text_and_table_only(benchmark_queries):

    previous_test_logs = set(list_log_numbers())   # For test_xx.json

    results = []
    answers = []

    rag_eval_path = "00-data/logs/rag_evaluation_logs.json"

    for q in benchmark_queries:
        print("\n" + "="*100)
        print(f"📊 {q['id']} — {q['name']}")
        print("="*100)

        user_query = q["query"]
        start_time = time.time()

        # Run CFO agent (no cache)
        response = query_cfo_agent(
            user_query,
            cache_threshold=0.85,
            use_cache=False
        )

        latency = round(time.time() - start_time, 2)
        answer_text = response["answer"].strip()

        # ==================================================
        # 1. GET NEW TEXT CONTEXT (from test_xx.json)
        # ==================================================
        current_test_logs = set(list_log_numbers())
        new_test_logs = list(current_test_logs - previous_test_logs)
        previous_test_logs = current_test_logs.copy()
        print (f" - New test logs: {new_test_logs}") 
    
        #text_chunks = []
        text_chunks: dict[str, List[str]] = {}
        for n in new_test_logs:
            log = load_log(n)
            #text_chunks.extend(extract_texts_from_log(log))  # your old text extractor
            extracted = extract_texts_from_log(log)  # your new text extractor
            text_chunks[extracted["expanded_query"]] = extracted["chunks"]

        #! DEBUGGING 
        print (f" - Extracted text chunks keys: {list(text_chunks.keys())}") 
        print (f" - Number of text chunks per key: {[len(v) for v in text_chunks.values()]}") 
        # ==================================================
        # 2. GET NEW TABLE CONTEXT (from rag_evaluation_logs.json)
        # ==================================================
        table_chunks = []
        with open(rag_eval_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Last appended log entry corresponds to THIS query
        latest_entry = json.loads(lines[-1])

        for ctx in latest_entry["contexts"]:
            if ctx.get("type") == "table" or ctx.get("subtype") == "financial_table":
                content = ctx.get("content", "").strip()
                if content:
                    table_chunks.append(content)

        # ==================================================
        # 3. COMBINE TEXT + TABLE ONLY
        # ==================================================
        #combined_context = text_chunks + table_chunks

        #if not combined_context:
            #combined_context = [""]

        # ==================================================
        # 4. CONTEXT RELEVANCE (COSINE + LLM)
        # ==================================================
        #ctx_cos = compute_context_relevance(user_query, combined_context)
        ctx_cos = compute_context_relevance(user_query, text_chunks, table_chunks) 

        print (f" - Context Relevance (cosine): {ctx_cos}") 

        #ctx_llm = context_relevance_score(user_query, "\n\n".join(combined_context))
        ctx_llm = context_relevance_score(user_query, text_chunks, table_chunks) 
        print (f" - Context Relevance (LLM): {ctx_llm}") 
        
        #! give 70% weight to llm since cosine may be low with sparse context 
        ctx_merged = 0.3 * ctx_cos + 0.7 * ctx_llm 
        #ctx_merged = (ctx_cos + ctx_llm) / 2
        print (f" - Context Relevance (Merged): {ctx_merged}") 


        # ==================================================
        # 5. ANSWER RELEVANCE
        # ==================================================
        ans_cos = compute_answer_relevance(user_query, answer_text)
        ans_llm = answer_relevance_score(user_query, answer_text)
        print (f" - Answer Relevance (cosine): {ans_cos}") 
        print (f" - Answer Relevance (LLM): {ans_llm}") 

        #ans_merged = (ans_cos + ans_llm) / 2
        ans_merged = 0.3 * ans_cos + 0.7 * ans_llm 
        # ==================================================
        # 6. FAITHFULNESS
        # ==================================================
        #faith = compute_faithfulness(answer_text, combined_context)
        faith = compute_faithfulness(answer_text, text_chunks, table_chunks) 
        print (f" - Faithfulness: {faith}") 
        faith_merged = faith  # no llm faithfulness for now 
        # faith_llm = faithfulness_score(answer_text, text_chunks, table_chunks) 
        # print (f" - Faithfulness (LLM): {faith_llm}") 
        # faith_merged = 0.3 * faith + 0.7 * faith_llm

        # ==================================================
        # 7. FINAL TRIAD MERGED
        # ==================================================
        triad_merged = float(np.mean([ctx_merged, ans_merged, faith_merged]))

        # ==================================================
        # 8. RECORD RESULTS
        # ==================================================
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

            "Faithfulness": round(faith_merged, 3),
            "Avg Triad (Merged)": round(triad_merged, 3),

            "Num Logs": len(new_test_logs),
            "Log Files": new_test_logs,
        })

        answers.append({
            "Query ID": q["id"],
            "Query Name": q["name"],
            "Full Answer": answer_text,
            "Sources": response.get("metadata", {})
        })

    save_run_logs(results, answers)

    return pd.DataFrame(results), pd.DataFrame(answers)


if __name__ == "__main__": 
    # Example benchmark queries
    # benchmark_queries = [
    #     {
    #         "id": "Q1",
    #         "name": "Gross Margin Trend",
    #         "query": "What is the gross margin trend for the last 3 years?"
    #     }
    # ]


    benchmark_queries = [
    {"id": 1, "name": "Gross Margin Trend", "query": "What is the gross margin trend for the last 3 years?"},
    {"id": 2, "name": "Operating Expenses YoY", "query": "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison."},
    {"id": 3, "name": "Operating Efficiency Ratio", "query": "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years."}
]

    # Run full RAG benchmark
    results_df, answers_df = run_full_rag_benchmark_text_and_table_only(benchmark_queries)

    print("\nFinal Results:")
    print(results_df) 
    print ("\nFinal Answers:") 
    print(answers_df) 