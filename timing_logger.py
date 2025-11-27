import os
import json
import re

def build_timing_json(question, callback, retrieval_time, rerank_time, total_time, cache_hits=False):
    
    # skip LLM timing if cache hit
    if cache_hits or callback is None:
        reasoning = 0.0
        generation = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
    else:
        durations = callback.llm_calls

        if len(durations) == 0:
            reasoning = 0.0
            generation = 0.0
        elif len(durations) == 1:
            reasoning = 0.0
            generation = durations[0]
        else:
            # Last call is generation, everything else is reasoning
            reasoning = sum(durations[:-1])
            generation = durations[-1]

        prompt_tokens = sum(callback.prompt_tokens)
        completion_tokens = sum(callback.completion_tokens)
        total_tokens = sum(callback.total_tokens)

    all_timings = {
        "Query": question,
        "T_ingest": None,
        "T_retrieve": retrieval_time,
        "T_rerank": rerank_time,
        "T_reason": reasoning,
        "T_generate": generation,
        "T_total": total_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_hit": cache_hits
    }

    dir_path = "00-data/logs/instru_log_testings"
    os.makedirs(dir_path, exist_ok=True)

    # Find existing files
    existing = [
        f for f in os.listdir(dir_path)
        if re.match(r"timings_(\d+)\.json$", f)
    ]

    # Determine next number to set as the file name
    if existing:
        nums = [int(re.findall(r"timings_(\d+)\.json$", f)[0]) for f in existing]
        next_num = max(nums) + 1
    else:
        next_num = 1

    file_name = f"timings_{next_num}.json"

    # Save
    with open(os.path.join(dir_path, file_name), "w") as f:
        json.dump(all_timings, f, indent=4)

    print("Saved timings to :", file_name)
    return all_timings
