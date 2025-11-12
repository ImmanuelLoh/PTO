import re
import json
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

def choose_sections_for_query(llm: ChatOpenAI, 
                              query: str, available_sections: list) -> list:
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
