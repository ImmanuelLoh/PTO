from dotenv import load_dotenv, find_dotenv
import re
import os, glob
import pdfplumber
import camelot
import pymupdf
import numpy as np
import pandas as pd
from pathlib import Path
import google.generativeai as genai
import time
import faiss, json
import collections
import fitz
import io
from PIL import Image
import pytesseract
from langchain_core.documents import Document
import chromadb
import gc
from openai import OpenAI


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

    header = (
        f"### Table {table_num} — {report}\n"
        f"**Page:** {page}\n\n"
    )

    try:
        # Convert to markdown table text
        table_md = df.to_markdown(index=False)
    except Exception:
        table_md = df.to_string(index=False)

    return header + table_md

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
        chunk = {
            'id': f"{source_name}_p{table_entry['page']}_t{table_entry['table_num']}",  # ← ADD THIS LINE
            'content': table_to_markdown(table_entry),
            'metadata': {
                'source': source_name,
                'page': table_entry['page'],
                'table_num': table_entry['table_num'],
                'type': 'financial_table',
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

def store_in_chromadb(chunks, collection_name, overwrite=False):
    """
    Store embedded chunks in ChromaDB.
    Assumes chunks already have 'embedding' field.
    """
    print(f"Storing in ChromaDB collection '{collection_name}'...\n")
    
    client = chromadb.PersistentClient(path="00-data/chromadb")
    
    # Handle existing collection
    if overwrite:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'.")
        except:
            pass
    
    # Create collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Store in batches
    total_stored = 0
    batch_size = 50
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1

        try: 
            collection.add(
                ids=[chunk['id'] for chunk in batch],
                embeddings=[chunk['embedding'] for chunk in batch],
                documents=[chunk['content'] for chunk in batch],
                metadatas=[chunk['metadata'] for chunk in batch]
            )
            total_stored += len(batch)
            print(f"   Batch {batch_num}/{num_batches}: stored {len(batch)} chunks (total: {total_stored}/{len(chunks)})")

            gc.collect()  # Force garbage collection to manage memory
            time.sleep(0.2)  # Small delay to avoid overwhelming ChromaDB
        except Exception as e:
            print(f"\n❌ Error storing batch {batch_num}: {str(e)}")
            raise
    
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB\n")
    return collection

def load_chromadb_collection(collection_name):
    """
    Load an existing ChromaDB collection.
    """
    client = chromadb.PersistentClient(path="00-data/chromadb")
    collection = client.get_collection(name=collection_name)
    
    print(f"✅ Loaded collection '{collection_name}'")
    print(f"   Total items: {collection.count()}\n")
    
    return collection

# def search_financial_tables(query_text, collection_name="financial_tables", top_k=5):
#     """
#     Search for relevant financial tables based on a text query.
    
#     Args:
#         query_text: Natural language query (e.g., "revenue growth", "operating expenses")
#         collection_name: Name of the ChromaDB collection
#         top_k: Number of results to return
    
#     Returns:
#         Dictionary with documents, metadatas, and distances
#     """
#     # Load collection
#     client = chromadb.PersistentClient(path="00-data/chromadb")
#     collection = client.get_collection(name=collection_name)
    
#     # Create query embedding
#     print(f"Searching for: '{query_text}'")
#     response = client_openai.embeddings.create(
#         model=EMBED_MODEL,
#         input=query_text
#     )
#     query_embedding = response.data[0].embedding
    
#     # Search ChromaDB
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )
    
#     print(f"Found {len(results['documents'][0])} results\n")
    
#     return results


# def display_search_results(results):
#     """
#     Pretty print search results.
#     """
#     for i, (doc, metadata, distance) in enumerate(zip(
#         results['documents'][0],
#         results['metadatas'][0],
#         results['distances'][0]
#     )):
#         print(f"{'='*60}")
#         print(f"Result {i+1} (Score: {1 - distance:.4f})")
#         print(f"{'='*60}")
#         print(f"Source: {metadata['source']}")
#         print(f"Page: {metadata.get('page', 'N/A')}")
#         print(f"Table: {metadata.get('table_number', 'N/A')}")
#         print(f"\nContent Preview:")
#         print(doc[:300] + "..." if len(doc) > 300 else doc)
#         print()

if __name__ == "__main__":   
    load_dotenv()
    
    # Check if key was loaded
    key = os.getenv('OPENAI_API_KEY')
    if key:
        print(f"Key loaded: {key[:15]}")
    else:
        print("No key found in environment!")

    COMPANY_NAME = "Google"
    DATA_DIR = "00-data"

    # table_chunks = []

    # for folder in ["annuals", "quarterlies"]:
    #     files = glob.glob(f"{DATA_DIR}/{folder}/*.pdf")
    #     print(f"{folder}: {len(files)} files")

    #     for pdf_file in files:
    #         print(f"Processing: {pdf_file}")
    #         chunks = extract_and_chunk_tables(pdf_file)
    #         table_chunks.extend(chunks)

    # print(f"\n✅ Total chunks: {len(table_chunks)}")

    # # Create embeddings for all chunks
    # embedded_table_chunks = create_embeddings(table_chunks)

    # # Save ONCE after processing all PDFs
    # store_in_chromadb(embedded_table_chunks, collection_name="financial_tables", overwrite=True)

    # Load existing collection
    load_chromadb_collection("financial_tables")

    # Example search
    query = "operating expenses and revenue growth for 2024"
    # results = search_financial_tables(query)
    # display_search_results(results)