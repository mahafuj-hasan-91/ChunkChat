"""
rag/pdf_loader.py
─────────────────
Handles PDF text extraction.

RAG Concept: "Document Ingestion"
  Before we can answer questions about a PDF, we must first convert it
  from binary format into raw text. This is Step 1 of the RAG pipeline.
  The quality of extraction directly affects the quality of answers.
"""

import os
import pdfplumber
import PyPDF2
from typing import Optional


def extract_text_pdfplumber(pdf_path: str) -> str:
    """
    Extract text from a PDF using pdfplumber (primary method).

    pdfplumber is better than PyPDF2 for complex PDFs with tables,
    multi-column layouts, and non-standard encodings.

    Args:
        pdf_path (str): Absolute or relative path to the PDF file.

    Returns:
        str: Full extracted text from all pages, joined by newlines.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        Exception: If text extraction fails.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    full_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # extract_text() returns None if a page has no text layer
                page_text = page.extract_text()
                if page_text:
                    # Tag each page for potential debugging/source tracking
                    full_text.append(f"[Page {page_num + 1}]\n{page_text}")

        return "\n\n".join(full_text)

    except Exception as e:
        raise Exception(f"pdfplumber extraction failed: {str(e)}")


def extract_text_pypdf2(pdf_path: str) -> str:
    """
    Fallback text extraction using PyPDF2.

    Used when pdfplumber fails (e.g., encrypted PDFs with no password,
    or certain older PDF formats).

    Args:
        pdf_path (str): Absolute or relative path to the PDF file.

    Returns:
        str: Extracted text, or empty string if extraction fails.
    """
    full_text = []

    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text.append(f"[Page {page_num + 1}]\n{page_text}")

        return "\n\n".join(full_text)

    except Exception as e:
        raise Exception(f"PyPDF2 extraction failed: {str(e)}")


def load_pdf(pdf_path: str) -> Optional[str]:
    """
    Main entry point for PDF loading with automatic fallback.

    Strategy:
      1. Try pdfplumber (handles most modern PDFs well)
      2. Fall back to PyPDF2 if pdfplumber fails
      3. Return None if both fail (caller handles this gracefully)

    Args:
        pdf_path (str): Path to the uploaded PDF file.

    Returns:
        Optional[str]: Extracted text if successful, None otherwise.
    """
    print(f"[PDF Loader] Loading: {pdf_path}")

    # --- Attempt 1: pdfplumber ---
    try:
        text = extract_text_pdfplumber(pdf_path)
        if text.strip():
            print(f"[PDF Loader] Success via pdfplumber. "
                  f"Extracted {len(text)} characters.")
            return text
        else:
            print("[PDF Loader] pdfplumber returned empty text. Trying fallback...")
    except Exception as e:
        print(f"[PDF Loader] pdfplumber failed: {e}. Trying PyPDF2...")

    # --- Attempt 2: PyPDF2 fallback ---
    try:
        text = extract_text_pypdf2(pdf_path)
        if text.strip():
            print(f"[PDF Loader] Success via PyPDF2. "
                  f"Extracted {len(text)} characters.")
            return text
        else:
            print("[PDF Loader] PyPDF2 also returned empty text.")
            return None
    except Exception as e:
        print(f"[PDF Loader] PyPDF2 also failed: {e}")
        return None


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Extract metadata from a PDF (title, author, page count, etc.).

    Useful for displaying document info in the UI and for debugging.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: Metadata dictionary with keys like 'pages', 'title', 'author'.
              Returns a minimal dict if extraction fails.
    """
    metadata = {"pages": 0, "title": "Unknown", "author": "Unknown"}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata["pages"] = len(pdf.pages)

            # pdfplumber exposes PDF metadata via .metadata
            if pdf.metadata:
                metadata["title"] = pdf.metadata.get("Title", "Unknown") or "Unknown"
                metadata["author"] = pdf.metadata.get("Author", "Unknown") or "Unknown"

    except Exception as e:
        print(f"[PDF Loader] Metadata extraction failed: {e}")

    return metadata
