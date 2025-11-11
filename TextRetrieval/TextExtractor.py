from dotenv import load_dotenv
import re
import os, glob
import pdfplumber
import openai
import time
import faiss, json
import collections

from sentence_transformers import util

from Embedding import embed_text_query, embed_text_passage 
from examples import SECTION_EXAMPLES 

COMPANY_NAME = "Google"




DATA_DIR = "00-data"

# Annual reports (10-Ks)
annual_files = glob.glob(f"{DATA_DIR}/annuals/*.pdf")
# # Quarterly reports (10-Qs)
quarterly_files = glob.glob(f"{DATA_DIR}/quarterlies/*.pdf")


# --- Helpers ---
def _normalize(s: str) -> str:
    s = (s or "").lower()
    # unify whitespace & quotes
    s = s.replace("\n", " ").replace("’", "'").replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s







# def classify_section(text):

#     SECTION_EMBS = {
#         sec: [embed_text_query(ex) for ex in examples]
#         for sec, examples in SECTION_EXAMPLES.items()
#         }
        
#     page_text = _normalize(text)
#     emb = embed_text_query(page_text)

#     scores = {
#         sec: max(util.cos_sim(emb, e).item() for e in embs)
#         for sec, embs in SECTION_EMBS.items()
#     }

#     best = max(scores, key=scores.get)
#     return best if scores[best] > 0.35 else "other"


def classify_section_hybrid(text, section_embs):

    emb = embed_text_query(_normalize(text))

    print (f"section_embs keys: {list(section_embs.keys())}") 
    print (section_embs)
    print ("==================================================================")
    scores = {
        sec: max((util.cos_sim(emb, e).item() for e in embs), default=0.0) 
    for sec, embs in section_embs.items()} 

    print (f"scores: {scores}")



    best_cosine_section = max(scores, key=scores.get)
    cosine_confidence = scores[best_cosine_section]
    print (f"Cosine best section: {best_cosine_section} with confidence {cosine_confidence}") 



    # Step 2: OpenAI model
    openai_prompt = f"""
    You are an expert financial analyst. Classify the section of this document.
    Sections: {list(section_embs.keys()) + ['other']}
    Text:
    {text}
    If none of the sections match and you deem it to be relatviely important, give a new section name.
    return STRICTLY the name with no explanation.
    """
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": openai_prompt}]
    )
    best_openai_section = response.choices[0].message.content.strip().lower()

    # Assign OpenAI confidence (simplified heuristic)
    openai_confidence = 0.8 if best_openai_section != "other" else 0.3

    # Step 3: Combine (tune these weights)
    α, β = 0.6, 0.4  # trust cosine slightly more
    final_scores = collections.defaultdict(float)

    # give weighted vote
    final_scores[best_cosine_section] += α * cosine_confidence
    final_scores[best_openai_section] += β * openai_confidence

    # pick winner
    best_final = max(final_scores, key=final_scores.get)
    return best_final



def extract_text_from_pdf():

    
    pdf_path = annual_files + quarterly_files

    print(f"Processing {len(pdf_path)} PDFs from all folders")
    print("PDF paths:", pdf_path[:3], "...")

    # keep track sections
    sections = {}
    output = {}


    # Step 1: cosine similarity model
    section_embs = {
        sec: [embed_text_query(ex) for ex in examples]
        for sec, examples in SECTION_EXAMPLES.items()
    }

    for pdfFile in pdf_path:
        pdf_name = os.path.basename(pdfFile)
        output[pdf_name] = {}

        print(f"\n=== Processing: {pdf_name} ===")

        # Step 1: extract raw text with pdfplumber
        with pdfplumber.open(pdfFile) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                section_text = classify_section_hybrid(text, section_embs)  # classifying page text only without tables
                if section_text not in section_embs:
                    section_embs[section_text] = [];
                sections[section_text] = sections.get(section_text, 0) + 1

            
                print(f"Page {i} → Text length: {len(text) if text else 0}")
                output[pdf_name][i] = {
                    "page_section": section_text,
                    "text": text,
                }

    print ("Section distribution:", sections)

    # Step 3: Create directory if it doesn't exist and dump to JSON
    output_path = f"{DATA_DIR}/optimizedText.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__": 
    load_dotenv() 
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    extract_text_from_pdf()
