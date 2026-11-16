
import re
import html
import unicodedata

import pandas as pd
from underthesea import word_tokenize


def clean_text(text):
    """
    Làm sạch văn bản tiếng Việt.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Chuẩn hóa Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # Chuyển chữ thường
    text = text.lower()

    # Giải mã HTML entities
    text = html.unescape(text)

    # Loại bỏ URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Loại bỏ email
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text
    )
    # Loại bỏ dấu câu còn sót lại

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )


    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def segment_text(text):
    """
    Tách từ tiếng Việt bằng Underthesea.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip()

    if not text:
        return ""

    return word_tokenize(
        text,
        format="text"
    )


def preprocess_and_segment(text):
    """
    Quy trình đầy đủ:
    Văn bản gốc -> Làm sạch -> Tách từ.
    """

    cleaned = clean_text(text)

    if not cleaned:
        return ""

    return segment_text(cleaned)
