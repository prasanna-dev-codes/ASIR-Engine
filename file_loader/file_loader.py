import os
from pypdf import PdfReader
from docx import Document
import pdfplumber
import fitz  # pymupdf


def load_pdf(file_path: str) -> dict:
    """
    Extract text from PDF using pymupdf for body text
    and pdfplumber for table detection.
    """
    text = ""
    total_pages = 0

    try:
        # --- Extract body text with pymupdf (better quality) ---
        doc = fitz.open(file_path)
        total_pages = len(doc)

        for page in doc:
            # "text" mode preserves layout better than default
            page_text = page.get_text("text")
            if page_text:
                text += page_text + "\n"

        doc.close()

        # --- Extract tables separately with pdfplumber ---
        # Tables come out garbled in body text extraction
        # so we pull them cleanly and append
        table_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        # filter None cells and join with separator
                        clean_row = [cell if cell else "" for cell in row]
                        table_text += " | ".join(clean_row) + "\n"
                    table_text += "\n"

        if table_text.strip():
            text += "\n\n--- Tables ---\n" + table_text

    except Exception as e:
        print(f"  ❌ Error reading PDF '{file_path}': {e}")

    return {
        "content": text,
        "metadata": {
            "total_pages": total_pages,
            "file_type": "pdf"
        }
    }


# -------- TXT LOADER --------
def load_txt(file_path: str) -> dict:
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        # some txt files are not utf-8, fallback encoding
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        except Exception as e:
            print(f"  ❌ Error reading TXT '{file_path}': {e}")
    except Exception as e:
        print(f"  ❌ Error reading TXT '{file_path}': {e}")

    return {
        "content": text,
        "metadata": {
            "file_type": "txt"
        }
    }

# -------- DOCX LOADER --------
def load_docx(file_path: str) -> dict:
    text = ""
    total_paragraphs = 0
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        total_paragraphs = len(paragraphs)
        text = "\n".join(paragraphs)
    except Exception as e:
        print(f"  ❌ Error reading DOCX '{file_path}': {e}")

    return {
        "content": text,
        "metadata": {
            "total_paragraphs": total_paragraphs,
            "file_type": "docx"
        }
    }


# -------- SINGLE FILE LOADER --------
def load_document(file_path: str) -> dict:
    """
    Load a single file and return its text content + metadata.
    Returns a dict with keys: file_name, file_path, content, metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    loaders = {
        ".pdf":  load_pdf,
        ".txt":  load_txt,
        ".docx": load_docx,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}")

    result = loaders[ext](file_path)

    # Warn if extraction returned empty text
    if not result["content"].strip():
        print(f"  ⚠️  Warning: No text extracted from '{file_name}'. File may be scanned/image-based.")

    return {
        "file_name": file_name,
        "file_path": file_path,
        "content":   result["content"],
        "metadata":  result["metadata"],
    }


# -------- FOLDER LOADER --------
def load_documents_from_folder(folder_path: str) -> list[dict]:
    """
    Load all supported files from a folder and its subfolders.
    Returns a list of dicts with keys: file_name, file_path, content, metadata
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    supported_extensions = {".pdf", ".txt", ".docx"}
    all_docs = []

    # os.walk goes into subfolders too unlike os.listdir
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()   # consistent order
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext not in supported_extensions:
                continue

            file_path = os.path.join(root, file)
            print(f"  📄 Loading: {file_path}")

            try:
                doc = load_document(file_path)
                all_docs.append(doc)
            except Exception as e:
                print(f"  ⚠️  Skipping '{file}': {e}")

    print(f"\n  ✅ Loaded {len(all_docs)} document(s) from '{folder_path}'")
    return all_docs