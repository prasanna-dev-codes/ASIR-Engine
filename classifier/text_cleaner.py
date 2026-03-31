"""
classifier/text_cleaner.py
==========================
Step 1 of the query classifier pipeline: text cleaning and normalisation.

WHAT THIS DOES:
  - Strips whitespace
  - Normalises unicode characters
  - Collapses multiple spaces
  - Removes control characters
  - Returns cleaned_text (lowercase, normalised) for internal analysis
  - Preserves raw_text unchanged for the LLM prompt

WHAT THIS DOES NOT DO:
  - No stopword removal (would destroy "between", "with" — relational signals)
  - No stemming or lemmatisation (changes meaning)
  - No spell correction (may corrupt technical terms)
  - No punctuation removal (question marks carry semantic meaning)
"""

import re
import unicodedata


def clean_query(raw_text: str) -> str:
    """
    Normalise raw query text for internal analysis.

    The cleaned text is used for:
      - spaCy linguistic analysis (Step 2)
      - Marker detection (comparative, relational)
      - Layer 1 signal aggregation

    The original raw_text is always preserved separately for:
      - The LLM prompt (LLMs perform better on natural-case text)
      - Logging and feedback records
      - User display

    Parameters
    ----------
    raw_text : str
        Exactly what the user typed. Not modified.

    Returns
    -------
    str
        Cleaned, normalised text ready for linguistic analysis.

    Examples
    --------
    >>> clean_query("  What  IS  photosynthesis?  ")
    'what is photosynthesis?'

    >>> clean_query("Compare Python\u2019s speed vs Java\u2019s speed")
    "compare python's speed vs java's speed"
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # Step 1: Normalise unicode to NFKC form.
    # This converts "curly quotes" → straight quotes,
    # "em dashes" → regular characters, etc.
    # NFKC is the right choice here — it normalises without
    # destroying meaning (unlike NFKD which strips diacritics).
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Replace curly/smart quotes with straight equivalents.
    # These commonly appear when users paste from Word or web pages.
    text = text.replace("\u2018", "'")   # left single quotation mark
    text = text.replace("\u2019", "'")   # right single quotation mark
    text = text.replace("\u201c", '"')   # left double quotation mark
    text = text.replace("\u201d", '"')   # right double quotation mark
    text = text.replace("\u2013", "-")   # en dash
    text = text.replace("\u2014", "-")   # em dash

    # Step 3: Remove control characters (non-printable ASCII)
    # These sometimes appear when text is copied from PDFs or terminals.
    # Keep: regular spaces (0x20), tabs (0x09), newlines (0x0A, 0x0D)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Step 4: Normalise whitespace.
    # Replace tabs and newlines with spaces first, then collapse runs.
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r" {2,}", " ", text)

    # Step 5: Strip leading and trailing whitespace.
    text = text.strip()

    # Step 6: Lowercase for internal analysis.
    # This ensures "Compare" and "compare" both match the marker "compare".
    # The original case is preserved in raw_text.
    text = text.lower()

    return text


def is_empty_query(raw_text: str) -> bool:
    """
    Return True if the query is empty or contains only whitespace.
    Used as a guard at the pipeline entry point.
    """
    return not raw_text or not raw_text.strip()


def get_word_count(text: str) -> int:
    """Count words in text by splitting on whitespace."""
    return len(text.split())


def get_char_count(text: str) -> int:
    """Count characters in raw text including spaces."""
    return len(text)
