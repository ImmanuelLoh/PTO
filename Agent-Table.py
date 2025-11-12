from dotenv import load_dotenv
import os, glob
import camelot
import pymupdf
import numpy as np
import pandas as pd
from pathlib import Path
import time
import faiss, json
from openai import OpenAI
from textwrap import shorten
from tabulate import tabulate

# Section classification helper
FINANCIAL_SECTIONS = {
    "income_statement": [
        "revenue", "cost of revenue", "operating expense", "operating income",
        "income from operations", "net income", "earnings per share", "diluted eps",
        "research and development", "sales and marketing", "general and administrative",
        "total costs and expenses", "provision for income taxes"
    ],
    "balance_sheet": [
        "assets", "liabilities", "shareholders equity", "stockholders equity",
        "current assets", "long term debt", "cash and cash equivalents",
        "accounts receivable", "property and equipment", "retained earnings"
    ],
    "cash_flow": [
        "cash flows", "operating activities", "investing activities",
        "financing activities", "cash provided by", "capital expenditures",
        "purchases of property", "free cash flow"
    ],
    "segment_info": [
        "segment", "geographic", "by region", "united states", "emea", "apac",
        "other americas", "constant currency", "revenues by geography"
    ],
    "summary_financial": [
        "consolidated revenues", "year ended december", "three months ended",
        "% change", "$ change", "prior period", "yoy", "year over year"
    ],
    "other": []
}

def classify_table_section(table_df):
    """
    Classify which financial section a table belongs to using keyword matching.
    """
    table_str = table_df.to_string().lower()
    
    section_scores = {}
    for section, keywords in FINANCIAL_SECTIONS.items():
        score = sum(1 for keyword in keywords if keyword in table_str)
        section_scores[section] = score
    
    best_section = max(section_scores, key=section_scores.get)
    return best_section if section_scores[best_section] > 0 else "other"

def extract_financial_terms(table_df):
    """
    Extract financial terms present in the table for better searchability.
    """
    table_str = table_df.to_string().lower()
    
    term_map = {
        'revenue': 'revenues',
        'consolidated revenue': 'consolidated revenues',
        'cost of revenue': 'cost of revenues',
        'operating expense': 'operating expenses',
        'research and development': 'R&D expenses',
        'sales and marketing': 'sales and marketing',
        'general and administrative': 'G&A expenses',
        'operating income': 'operating income',
        'income from operations': 'operating income',
        'gross profit': 'gross profit',
        'gross margin': 'gross margin',
        'net income': 'net income',
        'earnings per share': 'EPS',
        'diluted eps': 'diluted EPS',
        'operating margin': 'operating margin',
        'cash flow': 'cash flows',
        'capital expenditure': 'capital expenditures',
        'free cash flow': 'free cash flow'
    }
    
    found_terms = []
    for term, label in term_map.items():
        if term in table_str:
            found_terms.append(label)
    
    return list(set(found_terms))  # Remove duplicates


# Ingestion Pipeline
## Table Extraction
def extract_tables_from_page(pdf_path, page_num):
    """
    Extract tables from a PDF page by detecting colored header fills.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (1-indexed)

    Returns:
        List of DataFrames, one for each detected table
    """

    # Load the document and page
    doc = pymupdf.open(pdf_path)
    page = doc[page_num - 1]
    page_height = page.rect.height

    def is_nonwhite(rgb, thr=0.05):
        r, g, b = rgb
        return abs(1-r) + abs(1-g) + abs(1-b) > thr

    # --- 1. Get all fills ---
    fills = [
        (d["rect"], d["fill"]) for d in page.get_drawings()
        if d["type"] == "f" and d.get("fill")
    ]

    # --- 2. Keep only colored fills (blue/gray) ---
    colored = [f for f in fills if is_nonwhite(f[1]) and f[0].x1 - f[0].x0 > 100]
    colored.sort(key=lambda f: f[0].y0)

    # --- 3. Group colored fills into tables ---
    tables = []
    if colored:
        cur = [colored[0]]
        for f in colored[1:]:
            if abs(f[0].y0 - cur[-1][0].y1) < 25:  # stacked fills = same table
                cur.append(f)
            else:
                tables.append(cur)
                cur = [f]
        tables.append(cur)

    # --- 4. Process each detected table ---
    extracted_tables = []

    for idx, tgroup in enumerate(tables, 1):
        first_color = tgroup[0]
        y_bottom = max(f[0].y1 for f in tgroup) + 10
        y_top = first_color[0].y0

        # Find header region above this table
        header_y0 = y_top - 40
        x_left = min(f[0].x0 for f in tgroup)
        x_right = max(f[0].x1 for f in tgroup) + 100

        clip = pymupdf.Rect(x_left, header_y0 - 5, x_right, y_bottom + 5)

        try:
            # Convert clip to Camelot coords
            y1_cam = page_height - clip.y1
            y2_cam = page_height - clip.y0
            table_area = f"{clip.x0},{y1_cam},{clip.x1},{y2_cam}"

            tables_camelot = camelot.read_pdf(
                pdf_path,
                flavor="stream",
                table_areas=[table_area],
                pages=str(page_num)
            )

            if tables_camelot:
                df = tables_camelot[0].df
                extracted_tables.append({
                    'source': pdf_path,
                    'table_num': idx,
                    'page': page_num,
                    'clip': clip,
                    'dataframe': df
                })
            else:
                print(f"✗ Table {idx}: No table found")
        except Exception as e:
            print(f"✗ Table {idx} failed: {e}")

    doc.close()
    return extracted_tables

def extract_tables_from_pdf(pdf_path, pages=None):
    """
    Extract all tables from a PDF document.

    Args:
        pdf_path: Path to the PDF file
        pages: List of page numbers (1-indexed), or None for all pages
        use_camelot: Whether to fallback to Camelot if PyMuPDF fails

    Returns:
        List of dictionaries with table info and DataFrames
    """
    doc = pymupdf.open(pdf_path)

    if pages is None:
        pages = range(1, len(doc) + 1)

    all_tables = []

    for page_num in pages:
        tables = extract_tables_from_page(pdf_path, page_num)
        all_tables.extend(tables)

    doc.close()

    return all_tables

def table_to_markdown(table_entry):
    """
    Convert an extracted table dict into markdown with metadata headers.
    """
    df = table_entry["dataframe"]
    report = Path(table_entry.get("source", "Unknown")).stem
    page = table_entry["page"]
    table_num = table_entry["table_num"]

    # Classify table section and extract financial terms
    section = classify_table_section(df)
    terms = extract_financial_terms(df)

    header = f"### Table {table_num} - {report}\n"
    header += f"**Page:** {page}\n"
    header += f"**Section:** {section.replace('_', ' ').title()}\n"
    if terms:
        header += f"**Contains:** {', '.join(terms)}\n"
    header += "\n"

    try:
        # Convert to markdown table text
        table_md = df.to_markdown(index=False)
    except Exception:
        table_md = df.to_string(index=False)

    return header + table_md, section, terms

def extract_and_chunk_tables(pdf_path, pages=None):
    """
    Extract tables and prepare them directly as RAG chunks.
    No need to combine into a single markdown file!
    """
    # Extract tables
    all_tables = extract_tables_from_pdf(pdf_path, pages)

    # Convert each table to its own chunk
    chunks = []
    for table_entry in all_tables:
        source_name = Path(table_entry['source']).stem
        content, section, terms = table_to_markdown(table_entry)
        chunk = {
            'id': f"{source_name}_p{table_entry['page']}_t{table_entry['table_num']}",  # ← ADD THIS LINE
            'content': content,
            'metadata': {
                'source': source_name,
                'page': table_entry['page'],
                'table_num': table_entry['table_num'],
                'type': 'financial_table',
                'section': section,
                'financial_terms': terms
            }
        }
        chunks.append(chunk)

    return chunks

def create_embeddings(chunks, model="text-embedding-3-small"):
    """
    Create embeddings for chunks using OpenAI.
    Returns: chunks with 'embedding' field added
    """
    print(f"Creating embeddings with {model}...\n")
    print(f"   Total chunks: {len(chunks)}")
    
    client = OpenAI()

    # Split into batches of 50
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"   Processing batch {batch_num} of {total_batches}...")

        contents = [chunk['content'] for chunk in batch]
        
        # Embed batch
        response = client.embeddings.create(
            model=model,
            input=contents
        )
        
        # Add embeddings to chunks
        for j, chunk in enumerate(batch):
            chunk['embedding'] = response.data[j].embedding

        # Small delay to respect rate limits
        if i + batch_size < len(chunks):
                time.sleep(0.6) # Adjust as needed
    
    print(f"Created {len(chunks)} embeddings\n")
    return chunks

# FAISS
def store_in_faiss(embedded_chunks, faiss_index_path="faiss_table_index"):
    """
    Store embedded chunks in a FAISS index.
    """
    # Extract embeddings
    dimension = len(embedded_chunks[0]['embedding'])
    embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks]).astype('float32')

    # # Normalize embeddings for cosine similarity
    # faiss.normalize_L2(embeddings)

    # Create inner product index (cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, faiss_index_path)
    print(f"Stored {len(embedded_chunks)} vectors in FAISS index at '{faiss_index_path}'\n")

    return index

def load_faiss_index(faiss_index_path="faiss_table_index"):
    """
    Load a FAISS index from disk.
    """
    index = faiss.read_index(faiss_index_path)
    print(f"Loaded FAISS index from '{faiss_index_path}' with {index.ntotal} vectors\n")
    return index

def save_metadata_mapping(embedded_chunks, mapping_path="faiss_table_metadata.json"):
    """
    Save metadata mapping for FAISS index.
    Saves full chunk (id, content, metadata) without embeddings.
    """
    metadata = []
    for chunk in embedded_chunks:
        # Remove embedding field
        meta = {k: v for k, v in chunk.items() if k != 'embedding'}
        metadata.append(meta)

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata mapping for {len(embedded_chunks)} chunks to '{mapping_path}'\n")

def load_metadata_mapping(mapping_path="faiss_table_metadata.json"):
    """
    Load metadata mapping for FAISS index.
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        metadata_mapping = json.load(f)
    
    print(f"Loaded metadata mapping from '{mapping_path}' with {len(metadata_mapping)} entries\n")
    return metadata_mapping

def pretty_print_results(results, show_table=False):
    table_data = []
    for r in results:
        meta = r["metadata"]
        chunk_type = meta.get("type", "unknown")
        section = meta.get("section", "unknown")
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        table_num = meta.get("table_num", "?")
        score = f"{r['score']:.3f}"
        
        # shorten text for preview
        preview = shorten(r['text'], width=120, placeholder="…")
        table_data.append([r['rank'], score, chunk_type, section, source, page, table_num, preview])
    headers = ["Rank", "Score", "Type", "Section", "Source", "Page", "Table Number", "Preview"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))

    # print markdown tables
    if show_table:
        for r in results:
            if r["metadata"].get("type") == "financial_table" and r.get("markdown"):  # Changed "chunk_type" to "type"
                print(f"\nTable {r['metadata']['table_num']} from {r['metadata']['source']} (Page {r['metadata']['page']}):\n")
                print(r["markdown"])
                print("\n" + "-"*80 + "\n")

def search_tables(query, faiss_index_path, metadata_path, k=5):
    """
    Search for tables relevant to the query using FAISS index.

    Args:
        query_text: Natural language query
        faiss_index_path: Path to FAISS index
        metadata_path: Path to chunks metadata JSON
        top_k: Number of results to return
    
    Returns:
        List of results with chunk data and similarity scores
    """
    client = OpenAI()

    # Load FAISS index and metadata
    index = load_faiss_index(faiss_index_path)
    chunks_metadata = load_metadata_mapping(metadata_path)

    # Create query embedding
    print(f"Searching for: {query}\n")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = np.array(response.data[0].embedding).astype('float32')

    # Search FAISS
    distances, indices = index.search(query_embedding.reshape(1, -1), k)

    results = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        results.append({
            'rank': i + 1,
            'chunk': chunks_metadata[idx],
            'similarity_score': float(dist)
        })
    
    print(f"Found {len(results)} results\n")
    return results

def display_search_results(results):
    formatted_results = []
    for r in results:
        chunk = r['chunk']
        formatted_results.append({
            'rank': r['rank'],
            'score': r['similarity_score'],
            'text': chunk['content'],
            'metadata': chunk['metadata'],
            'markdown': chunk['content'] if chunk['metadata'].get('type') == 'financial_table' else None
        })
    
    pretty_print_results(formatted_results, show_table=True)

def answer_query_with_rag(query, faiss_index_path, metadata_path, k=3):
    """
    Full RAG: Retrieve relevant tables + Generate answer with LLM
    """
    client = OpenAI()
    
    # 1. RETRIEVE relevant tables
    results = search_tables(query, faiss_index_path, metadata_path, k)
    
    # 2. AUGMENT - Build context from retrieved tables
    context = "\n\n".join([
        f"Table from {r['chunk']['metadata']['source']} (Page {r['chunk']['metadata']['page']}):\n{r['chunk']['content']}"
        for r in results
    ])
    
    # 3. GENERATE - Use LLM to answer based on context
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": 
                "You are a financial analyst assistant. Answer questions based ONLY on the provided financial tables. "
                "If the requested information is not in the tables, clearly state that the data is unavailable. "
                "Always cite which table and page your answer comes from."
                    },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ], 
        temperature=0.2  # Lower temperature for more consistent financial analysis
    )
    
    return response.choices[0].message.content

def run_benchmark_queries(faiss_index_path, metadata_path):
    """
    Run standardized financial analysis queries.
    """
    client = OpenAI()
    
    queries = {
        "gross_margin": {
            "query": "Report the Gross Margin over the last 5 quarters, with values.",
            "formula": "Gross Margin % = (Gross Profit ÷ Revenue) × 100\nGross Profit = Revenue - Cost of Revenues"
        },
        "operating_expenses_yoy": {
            "query": "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.",
            "formula": "Operating Expenses = R&D + Sales & Marketing + General & Administrative"
        },
        "operating_efficiency": {
            "query": "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.",
            "formula": "Operating Efficiency Ratio = Operating Expenses ÷ Operating Income × 100\nOperating Income = Gross Profit - Operating Expenses"
        },
        "net_interest_margin": {
            "query": "Report Net Interest Margin (NIM) trend over last 5 quarters, values and 1-2 lines of explanation.",
            "formula": "For banks only"
        },
        "cost_to_income": {
            "query": "Show Cost-to-Income Ratio (CTI) for last 3 years; show working + implications.",
            "formula": "Expected: Operating Income & Opex lines"
        }
    }
    
    results = {}
    
    for name, info in queries.items():
        print("\n" + "="*80)
        print(f"BENCHMARK: {name.replace('_', ' ').title()}")
        print("="*80)
        print(f"Formula: {info['formula']}\n")
        
        # Retrieve relevant tables
        search_results = search_tables(
            info['query'],
            faiss_index_path,
            metadata_path,
            k=15
        )
        
        # Build context
        context = "\n\n".join([
            f"Table from {r['chunk']['metadata']['source']} (Page {r['chunk']['metadata']['page']}):\n{r['chunk']['content']}"
            for r in search_results
        ])
        
        # Generate answer with enhanced prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a financial analyst assistant. Analyze the provided financial tables and:\n"
                        "1. Extract the requested metrics\n"
                        "2. Show calculations and formulas used\n"
                        "3. Present data in table format when appropriate\n"
                        "4. Provide brief analysis and insights\n"
                        "5. Cite which tables you used\n"
                        "If data is unavailable, clearly state what's missing."
                    )
                },
                {
                    "role": "user", 
                    "content": f"Formula/Context:\n{info['formula']}\n\nFinancial Tables:\n{context}\n\nQuery: {info['query']}\n\nProvide detailed analysis with calculations:"
                }
            ],
            temperature=0.2  # Lower temperature for more consistent financial analysis
        )
        
        answer = response.choices[0].message.content
        results[name] = {
            "query": info['query'],
            "answer": answer,
            "sources": [r['chunk']['metadata']['source'] for r in search_results[:3]]
        }
        
        print(f"Query: {info['query']}\n")
        print(f"Retrieved {len(search_results)} tables")
        print(f"Answer:\n{answer}\n")
        print(f"Sources: {', '.join(results[name]['sources'])}")
    
    return results

if __name__ == "__main__":   
    load_dotenv()
    
    # Check if key was loaded
    key = os.getenv('OPENAI_API_KEY')

    COMPANY_NAME = "Google"
    DATA_DIR = "00-data"

    # ============================================================================
    # STEP 1: INGESTION - Extract tables and create embeddings
    # ============================================================================
    # table_chunks = []

    # for folder in ["annuals", "quarterlies"]:
    #     files = glob.glob(f"{DATA_DIR}/{folder}/*.pdf")
    #     print(f"{folder}: {len(files)} files")

    #     for pdf_file in files:
    #         print(f"Processing: {pdf_file}")
    #         chunks = extract_and_chunk_tables(pdf_file)
    #         table_chunks.extend(chunks)

    # print(f"\nTotal chunks: {len(table_chunks)}")

    # # Create embeddings for all chunks
    # embedded_table_chunks = create_embeddings(table_chunks)

    # # Store in FAISS and save metadata mapping
    # faiss_index = store_in_faiss(
    #     embedded_table_chunks, 
    #     faiss_index_path=f"{DATA_DIR}/faiss_table_index"
    #     )
    # save_metadata_mapping(
    #     embedded_table_chunks, 
    #     mapping_path=f"{DATA_DIR}/faiss_table_metadata.json"
    #     )

    # # Example search
    # print("\n" + "="*80)
    # print("EXAMPLE 1: Semantic Search (Retrieval Only)")
    # print("="*80)
    # # query1 = "operating expenses and revenue growth for 2024"
    # query1 = "What were the operating expenses in Q3 2024?"
    # results = search_tables(
    #     query1,
    #     faiss_index_path=f"{DATA_DIR}/faiss_table_index",
    #     metadata_path=f"{DATA_DIR}/faiss_table_metadata.json",
    #     k=5
    # )
    # display_search_results(results)

    # # Full RAG with LLM answer
    # print("\n" + "="*80)
    # print("EXAMPLE 2: Full RAG (Retrieval + Generation)")
    # print("="*80)
    # query2 = "What were the operating expenses in Q3 2024?"
    # answer = answer_query_with_rag(
    #     query2,
    #     faiss_index_path=f"{DATA_DIR}/faiss_table_index",
    #     metadata_path=f"{DATA_DIR}/faiss_table_metadata.json",
    #     k=3
    # )
    # print(f"\nQuery: {query2}")
    # print(f"\nAnswer:\n{answer}")

    # ============================================================================
    # STEP 2: BENCHMARK QUERIES - Run standardized financial analysis
    # ============================================================================
    print("\n" + "="*80)
    print("RUNNING BENCHMARK QUERIES")
    print("="*80)

    benchmark_results = run_benchmark_queries(
        faiss_index_path=f"{DATA_DIR}/faiss_table_index",
        metadata_path=f"{DATA_DIR}/faiss_table_metadata.json"
    )
    
    # Optionally save results
    with open(f"{DATA_DIR}/benchmark_results.json", 'w') as f:
        # Remove complex objects for JSON serialization
        simplified_results = {
            k: {
                "query": v["query"], 
                "answer": v["answer"], 
                "sources": v["sources"],
                # "num_tables": v["num_tables_retrieved"]
            }
            for k, v in benchmark_results.items()
        }
        json.dump(simplified_results, f, indent=2)
    
    print("\nBenchmark results saved to benchmark_results.json")
