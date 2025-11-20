import re
import json
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

import asyncio
from openai import AsyncOpenAI


def choose_sections_for_query(llm: ChatOpenAI, 
                              query: str, 
                              available_sections: list) -> list:
    """Internal helper — not a tool."""
    sec_list = ", ".join(available_sections)
    prompt = f"""
    You are a financial data retrieval router.
    Given the user's question and the available 10-Q sections, 
    select the most relevant section(s) to search for an answer.

    Available sections: {sec_list}
    User question: {query}

    Return a JSON array of section names from the list above. Strictly start with '[' and end with ']'.
    Example output: ["income_statement", "balance_sheet"]
    """

    response = llm.invoke([SystemMessage(content=prompt)])

    try:
        selected = json.loads(response.content)
        if isinstance(selected, list):
            return [sec for sec in selected if sec in available_sections]
        return []
    except Exception:
        return []
    
def expand_query_for_retrieval(query: str,
                               llm: ChatOpenAI
                               ) -> str:
    """Internal helper — not a tool."""
    prompt = f"""
    Expand the following financial query to improve retrieval from SEC filings.

    Guidelines:
    - Include synonyms (e.g., "operating expenses" → "SG&A", "total expenses").
    - Include both annual and quarterly phrasing (e.g., “fiscal year”, “quarter ended”).
    - Include context like "Consolidated Statements of Income", "Statements of Operations".
    - Keep it short but keyword-dense (2–3 sentences).

    User query: "{query}"
    """
    response = llm.invoke(prompt)
    return response.content.strip()



def determine_k(query: str) -> int:
    q = query.lower()

    # Multi-year, YoY, trends, comparisons → highest recall
    if any(keyword in q for keyword in [
        "last", "past", "previous", "historical",
        "trend", "yoy", "year over year",
        "3 year", "5 year", "compare", "comparison",
        "growth", "decline", "change"
    ]):
        return 40   # full 10-K table coverage

    # Operating expense components or cost structure → need many pages
    if any(keyword in q for keyword in [
        "operating expense", "opex",
        "r&d", "research", 
        "sales and marketing", "s&m",
        "general and administrative", "g&a",
        "cost of revenues", "tac", "margin"
    ]):
        return 35

    # Single-year lookup, but still numeric → 10-K tables are far away
    if any(str(year) in q for year in range(2018, 2030)):
        return 25

    # Section name queries (e.g., "md&a", "liquidity", "risk factors")
    if any(keyword in q for keyword in ["md&a", "risk", "liquidity"]):
        return 20

    # Simple text or conceptual questions (rare in your app)
    return 10


def safe_json_loads(s):
    """Tries to parse JSON even if it has weird hidden chars or minor formatting issues."""
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            cleaned = (
                s.strip("` \n\t'\"")          # remove outer quotes / markdown ticks / newlines
                .replace("'", '"')            # replace single quotes with double
                .replace("\ufeff", "")        # remove BOM marker
            )
            # remove trailing commas or stray markdown
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            return json.loads(cleaned)
        except Exception as e:
            print(f"[ERROR] safe_json_loads still failed: {e}")
            print(f"[DEBUG] Cleaned candidate: {repr(cleaned)}")
            return None


EXTRACTION_CACHE = {}
llm_async = AsyncOpenAI()

async def extract_financial_values_async(text: str, source: str):
    """
    ASYNC + CACHED version of financial extraction.
    """

    # ---- CACHING ----
    cache_key = f"{source}:{hash(text)}"
    if cache_key in EXTRACTION_CACHE:
        print (f"[CACHE HIT] extract_financial_values_async for {source}") 
        return EXTRACTION_CACHE[cache_key]

    print (f"[CACHE MISS] extract_financial_values_async for {source}") 
    prompt = f"""
You are a financial data extractor.

From the text below, extract ALL financial line items with amounts.
Return ONLY a JSON array of objects with this schema:

[
  {{
    "component": string,
    "year": number,
    "value": number,
    "source": "{source}"
  }}
]

Extraction rules:
- A "component" is any labeled financial line item (e.g., "Research and development", "General and administrative").
- Extract every number that represents money.
- Infer the year ONLY when the table shows multiple years (e.g., "2022 2023 2024" columns). If year cannot be determined, omit the row.
- Values must be numeric integers only (no commas).
- Use the provided source string unchanged for every row.

TEXT TO EXTRACT FROM:
{text}
"""

    # ---- ASYNC LLM CALL ----
    try:
        response = await llm_async.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        result_text = response.choices[0].message.content

        rows = json.loads(result_text)

        # save to cache
        EXTRACTION_CACHE[cache_key] = rows
        return rows

    except Exception as e:
        print(f"[ERROR] extract_financial_values_async failed for {source}")
        print(f"Reason: {e}")
        return []
    


def extract_financial_values(text: str, source: str):
    """
    Convert raw financial text into structured rows usable for tables + analytics.
    Extracts (component, year, value, source) using the LLM.
    """

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = f"""
You are a financial data extractor.

From the text below, extract ALL financial line items with amounts.
Return ONLY a JSON array of objects with this schema:

[
  {{
    "component": string,
    "year": number,
    "value": number,
    "source": "{source}"
  }}
]

Extraction rules:
- A "component" is any labeled financial line item (e.g., "Research and development", "General and administrative").
- Extract every number that represents money.
- Infer the year ONLY when the table shows multiple years (e.g., "2022 2023 2024" columns). If year cannot be determined, omit the row.
- Values must be numeric integers only (no commas).
- Use the provided source string unchanged for every row.

TEXT TO EXTRACT FROM:
{text}
"""

    result = llm.invoke(prompt).content
    try:
        return json.loads(result)
    except: 
        print(f"[ERROR] extract_financial_values failed to parse JSON: {result}") 
        return [] 