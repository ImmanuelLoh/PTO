import os
import fitz
import io
from PIL import Image

def extract_slides_fitz(pdf_path, output_dir):
    """
    Extract each slide as a PNG. One slide becomes one image.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf = fitz.open(pdf_path)

    print(f"[SlidesExtractor] Loaded {pdf_path} ({len(pdf)} pages)")

    for i, page in enumerate(pdf, start=1):
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        out_path = os.path.join(output_dir, f"slide_{i:02d}.png")
        img.save(out_path, "PNG")

    print(f"[SlidesExtractor] Extracted {len(pdf)} slides")
