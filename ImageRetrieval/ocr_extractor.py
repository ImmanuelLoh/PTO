import os
from PIL import Image
import pytesseract
from langchain_core.documents import Document

# Update this if your tesseract install is elsewhere
# Downloaded from https://github.com/UB-Mannheim/tesseract/wiki -> Download "tesseract-ocr-w64-setup-5.5.0.20241111.exe (64 bit)"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_folder(folder, label):
    """
    Convert each PNG slide into a LangChain Document.
    No chunking is done — one slide = one document.
    """
    docs = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".png"):
            path = os.path.join(folder, fname)
            text = pytesseract.image_to_string(Image.open(path))

            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"image_path": path, "source_report": label},
                    )
                )

    print(f"[OCR] Loaded {len(docs)} OCR docs from {folder}")
    return docs
