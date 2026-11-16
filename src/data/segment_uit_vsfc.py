
from pathlib import Path

import pandas as pd
from underthesea import word_tokenize

# =====================================
# 1. CẤU HÌNH
# =====================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "alternative" /
    "processed" / "uit_vsfc_preprocessed.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "alternative" /
    "processed" / "uit_vsfc_segmented.csv"
)


# =====================================
# 2. HÀM TÁCH TỪ
# =====================================

def segment_text(text):
    if pd.isna(text):
        return ""

    # Tách từ tiếng Việt
    return word_tokenize(
        str(text),
        format="text"
    )


# =====================================
# 3. ĐỌC VÀ XỬ LÝ DỮ LIỆU
# =====================================

def main():

    print("Đang đọc dữ liệu UIT-VSFC...")

    df = pd.read_csv(INPUT_FILE)

    if "clean_comments" not in df.columns:
        raise ValueError(
            "Không tìm thấy cột clean_comments."
        )

    print("Tổng số mẫu:", len(df))

    print("\nĐang thực hiện Word Segmentation...")

    df["segmented_comments"] = (
        df["clean_comments"].apply(segment_text)
    )

    # Thống kê số từ sau tách
    df["segmented_word_count"] = (
        df["segmented_comments"]
        .str.split()
        .str.len()
    )

    # Lưu kết quả
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nHOÀN THÀNH TÁCH TỪ")

    print("\nVí dụ trước và sau tách từ:")

    for _, row in df.head(10).iterrows():

        print("\nVăn bản gốc:")
        print(row["comments"])

        print("\nSau làm sạch:")
        print(row["clean_comments"])

        print("\nSau tách từ:")
        print(row["segmented_comments"])

        print("-" * 60)

    print("\nThống kê số từ sau tách:")

    print(
        df["segmented_word_count"].describe()
    )

    print("\nFile kết quả:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
