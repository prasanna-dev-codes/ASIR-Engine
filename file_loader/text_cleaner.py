import os
import re


def clean_text(text: str) -> str:

    # --- Basic garbage removal ---
    text = text.replace("\x00", "")
    text = text.replace("\uf0b7", "")
    text = text.replace("\uf020", " ")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)

    # --- Normalize line endings ---
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # --- Remove arXiv stamp lines only ---
    text = re.sub(r"arXiv:\S+.*\n?", "", text)

    # --- Remove lines that are ONLY a number (page numbers) ---
    # This only removes lines like "1" or "25" or " 3 "
    # It will NOT remove "Table 10" or "1 Introduction" 
    lines = text.split("\n")
    lines = [l for l in lines if not re.fullmatch(r"\s*\d{1,3}\s*", l)]

    # --- Remove figure fragment lines (short lines with only numbers like "11 12" or "53 54") ---
    lines = [l for l in lines if not re.fullmatch(r"[\s\d]+", l)]

    # --- Collapse multiple blank lines into one ---
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # --- Fix multiple spaces within lines ---
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

def save_to_txt(text: str, output_path: str) -> None:
    """
    Save cleaned text to a .txt file.
    Creates output folder automatically if it doesn't exist.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)                # create folder if missing

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  ✅ Saved to: {output_path}")