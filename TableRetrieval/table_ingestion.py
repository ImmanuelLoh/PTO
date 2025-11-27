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
            'id': f"{source_name}_p{table_entry['page']}_t{table_entry['table_num']}",
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

# =============================================================================
# EMBEDDINGS + FAISS
# =============================================================================

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

def store_in_faiss(embedded_chunks, faiss_index_path="faiss_table_index"):
    """
    Store embedded chunks in a FAISS index.
    """
    # Extract embeddings
    dimension = len(embedded_chunks[0]['embedding'])
    embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks]).astype('float32')

    # Create inner product index (cosine similarity after normalization)
    # index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexFlatL2(dimension)
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

def stage1_extract_and_save():
    """
    Extract all tables from PDFs and save to JSON.
    NO embeddings created yet - just pure extraction.
    """
    load_dotenv()
    
    COMPANY_NAME = "Google"
    DATA_DIR = "00-data"
    
    print("="*80)
    print("STAGE 1: EXTRACTING TABLES FROM PDFs")
    print("="*80)
    print()
    
    table_chunks = []
    
    for folder in ["annuals", "quarterlies"]:
        files = glob.glob(f"{DATA_DIR}/{folder}/*.pdf")
        print(f"\n{folder}: {len(files)} files")
        
        for pdf_file in files:
            print(f"  Processing: {pdf_file}")
            chunks = extract_and_chunk_tables(pdf_file)
            table_chunks.extend(chunks)
            print(f"    → Extracted {len(chunks)} tables")
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {len(table_chunks)} table chunks extracted")
    print(f"{'='*80}\n")
    
    # Save to JSON for inspection
    output_file = f"{DATA_DIR}/extracted_tables.json"
    
    print(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(table_chunks, f, indent=2)
    
    return output_file, len(table_chunks)

def stage2_create_embeddings(json_file):
    """
    Load extracted tables from JSON and create embeddings.
    This is where you spend OpenAI tokens.
    """
    load_dotenv()
    
    DATA_DIR = "00-data"
    
    print("="*80)
    print("STAGE 2: CREATING EMBEDDINGS")
    print("="*80)
    print()
    
    # Load the extracted tables
    print(f"Loading: {json_file}")
    with open(json_file, 'r') as f:
        table_chunks = json.load(f)
    
    print(f"Loaded {len(table_chunks)} tables")
    
    print("\nCreating embeddings...")
    embedded_chunks = create_embeddings(table_chunks)
    
    print("\nStoring in FAISS...")
    faiss_index = store_in_faiss(
        embedded_chunks,
        faiss_index_path=f"{DATA_DIR}/faiss_table_index"
    )
    
    print("\nSaving metadata...")
    save_metadata_mapping(
        embedded_chunks,
        mapping_path=f"{DATA_DIR}/faiss_table_metadata.json"
    )
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print("="*80)
    print(f"FAISS index: {DATA_DIR}/faiss_table_index")
    print(f"Metadata: {DATA_DIR}/faiss_table_metadata.json")
    print()