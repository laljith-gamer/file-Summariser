import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re

# Set tesseract path if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_text(text: str) -> str:
    """Remove unwanted symbols, fix spacing and line breaks."""
    # Remove strange OCR artifacts
    text = re.sub(r"[}{)(]+", " ", text)   # remove stray brackets
    text = re.sub(r"[~^_]+", " ", text)    # remove ~, ^, _
    text = re.sub(r"\s{2,}", " ", text)    # collapse multiple spaces
    text = re.sub(r"(\n\s*){2,}", "\n\n", text)  # collapse too many blank lines
    return text.strip()


def extract_text_from_pdf(pdf_path, use_ocr=True):
    doc = fitz.open(pdf_path)
    full_text = ""

    for i, page in enumerate(doc):
        text = page.get_text("text")

        # If text looks empty/garbled, fallback to OCR
        if use_ocr and (not text.strip() or "�" in text):
            full_text += f"\npage_{i+1}:\n"
            print(f"[INFO] Running OCR on page {i+1}...")
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="eng+hin")

        # Clean the extracted text
        text = clean_text(text)
        full_text += text + "\n\n"

    return full_text.strip()


if __name__ == "__main__":
    pdf_file = "example2.pdf"
    output_file = "extracted_text_clean.txt"

    extracted_text = extract_text_from_pdf(pdf_file, use_ocr=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"\n✅ Extraction complete! Cleaned text saved to {output_file}")
