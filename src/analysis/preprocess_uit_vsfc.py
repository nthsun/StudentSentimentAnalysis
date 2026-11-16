
from pathlib import Path
import re
import html
import unicodedata

import pandas as pd

# =====================================
# 1. CẤU HÌNH
# =====================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "alternative" / "uit_vsfc_1000.csv"
)

OUTPUT_DIR = ROOT / "data" / "alternative" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}


# =====================================
# 2. HÀM LÀM SẠCH VĂN BẢN
# =====================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Chuẩn hóa Unicode tiếng Việt
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

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================
# 3. ĐỌC DỮ LIỆU
# =====================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "comments",
        "sentiments",
        "sentiment_label"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Thiếu cột dữ liệu: {required_columns}"
        )

    print("Số mẫu ban đầu:", len(df))

    return df


# =====================================
# 4. XỬ LÝ DỮ LIỆU
# =====================================

def preprocess_data(df):
    df = df.copy()

    # Loại bỏ dòng thiếu dữ liệu
    df = df.dropna(
        subset=["comments", "sentiments"]
    )

    # Tạo cột văn bản đã tiền xử lý
    df["clean_comments"] = (
        df["comments"].apply(clean_text)
    )

    # Loại bỏ văn bản rỗng sau xử lý
    df = df[
        df["clean_comments"].str.strip() != ""
    ].copy()

    # Kiểm tra nhãn
    df["sentiments"] = df["sentiments"].astype(int)

    if not set(df["sentiments"].unique()).issubset(
        LABEL_NAMES.keys()
    ):
        raise ValueError("Phát hiện nhãn không hợp lệ.")

    # Loại bỏ các dòng trùng hoàn toàn
    before = len(df)

    df = df.drop_duplicates(
        subset=["clean_comments", "sentiments"]
    ).copy()

    removed_duplicates = before - len(df)

    # Tính số từ sau tiền xử lý
    df["processed_word_count"] = (
        df["clean_comments"]
        .str.split()
        .str.len()
    )

    # Kiểm tra câu có cùng nội dung nhưng khác nhãn
    conflicting = (
        df.groupby("clean_comments")["sentiments"]
        .nunique()
    )

    conflict_count = int((conflicting > 1).sum())

    print("\nKẾT QUẢ TIỀN XỬ LÝ")
    print("Số mẫu sau xử lý:", len(df))
    print("Số dòng trùng đã loại:", removed_duplicates)
    print("Số nội dung có nhãn mâu thuẫn:", conflict_count)

    return df


# =====================================
# 5. LƯU DỮ LIỆU
# =====================================

def save_data(df):
    output_file = OUTPUT_DIR / "uit_vsfc_preprocessed.csv"

    columns = [
        "comments",
        "clean_comments",
        "sentiments",
        "sentiment_label",
        "processed_word_count"
    ]

    df[columns].to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Lưu phân bố nhãn sau tiền xử lý
    distribution = (
        df["sentiment_label"]
        .value_counts()
        .rename_axis("sentiment_label")
        .reset_index(name="count")
    )

    distribution.to_csv(
        OUTPUT_DIR / "label_distribution_after.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPhân bố nhãn sau tiền xử lý:")
    print(distribution.to_string(index=False))

    print("\n5 mẫu sau tiền xử lý:")
    print(
        df[columns]
        .head()
        .to_string(index=False)
    )

    print("\nĐã lưu dữ liệu tại:")
    print(output_file)


# =====================================
# 6. CHƯƠNG TRÌNH CHÍNH
# =====================================

def main():
    try:
        df = load_data()
        df = preprocess_data(df)
        save_data(df)

        print("\nHOÀN THÀNH TIỀN XỬ LÝ UIT-VSFC")

    except Exception as error:
        print("\nĐÃ XẢY RA LỖI:")
        print(error)


if __name__ == "__main__":
    main()
