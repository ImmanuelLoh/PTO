import pdfplumber
import camelot
import json
import os


output = {}

pdf_path = "Google/goog-10-q-q2-2025.pdf" 

# Step 1: extract raw text with pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()

        # Step 2: extract tables with Camelot (try lattice first, then stream)
        tables_lattice = camelot.read_pdf(pdf_path, pages=str(i), flavor="lattice")
        tables_stream = camelot.read_pdf(pdf_path, pages=str(i), flavor="stream")

        tables = []

        if tables_lattice.n > 0:
            for t in tables_lattice:
                tables.append(t.df.values.tolist())  # convert DataFrame → list of rows
        elif tables_stream.n > 0:
            for t in tables_stream:
                tables.append(t.df.values.tolist())

        print(f"Page {i} → Text length: {len(text) if text else 0}, Tables found: {len(tables)}")

        output[i] = {
            "text": text,
            "tables": tables
        }



# Step 3: Create directory if it doesn't exist and dump to JSON
output_path = "Google/data/goog-10-q-q2-2025.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f: 
    json.dump(output, f, indent=4)