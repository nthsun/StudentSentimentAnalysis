
from pathlib import Path

import pandas as pd
from datasets import load_dataset

TOTAL_SAMPLES = 1000
RANDOM_STATE = 42

# Đường dẫn thư mục dự án
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "alternative"


LABEL_TO_ID = {
    "negative": 0,
    "positive": 1,
    "neutral": 2
}

LABEL_NAMES = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}


def download_dataset():
    print("Đang tải UIT-VSFC từ Hugging Face...")

    dataset = load_dataset("ura-hcmut/UIT-VSFC")

    print("\nThông tin các tập dữ liệu:")
    for split_name, split_data in dataset.items():
        print(f"- {split_name}: {len(split_data)} mẫu")

    # Chỉ sử dụng tập train để lấy 1.000 mẫu thử nghiệm
    df = pd.DataFrame(dataset["train"])

    print("\nCác cột dữ liệu:", df.columns.tolist())

    return df


# ==============================
# 3. LÀM SẠCH DỮ LIỆU CƠ BẢN
# ==============================

def clean_data(df):
    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset thiếu cột cần thiết: {required_columns}. "
            f"Các cột hiện tại: {df.columns.tolist()}"
        )

    df = df[["text", "label"]].copy()

    # Loại bỏ dữ liệu thiếu
    df = df.dropna(subset=["text", "label"])

    # Chuẩn hóa văn bản và nhãn
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Loại bỏ bình luận rỗng
    df = df[df["text"] != ""]

    # Kiểm tra nhãn hợp lệ
    actual_labels = set(df["label"].unique())
    expected_labels = set(LABEL_TO_ID.keys())

    if actual_labels != expected_labels:
        raise ValueError(
            f"Nhãn dữ liệu không hợp lệ.\n"
            f"Nhãn thực tế: {actual_labels}\n"
            f"Nhãn mong đợi: {expected_labels}"
        )

    return df


# ==============================
# 4. LẤY 1.000 MẪU CÂN BẰNG
# ==============================

def sample_balanced_data(df):
    print("\nPhân bố nhãn ban đầu:")
    print(df["label"].value_counts())

    # 1.000 mẫu: 333 Negative, 333 Positive, 334 Neutral
    samples_per_class = TOTAL_SAMPLES // 3

    sample_counts = {
        "negative": samples_per_class,
        "positive": samples_per_class,
        "neutral": TOTAL_SAMPLES - 2 * samples_per_class
    }

    sampled_groups = []

    for label, count in sample_counts.items():
        class_data = df[df["label"] == label]

        if len(class_data) < count:
            raise ValueError(
                f"Lớp {label} chỉ có {len(class_data)} mẫu, "
                f"không đủ để lấy {count} mẫu."
            )

        sampled = class_data.sample(
            n=count,
            random_state=RANDOM_STATE
        )

        sampled_groups.append(sampled)

    # Gộp và xáo trộn dữ liệu
    df_final = pd.concat(
        sampled_groups,
        ignore_index=True
    )

    df_final = df_final.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return df_final


# ==============================
# 5. CHUẨN HÓA NHÃN VÀ CỘT
# ==============================

def prepare_output(df):
    df_final = df.rename(columns={
        "text": "comments"
    }).copy()

    # Chuyển nhãn chuỗi thành nhãn số
    df_final["sentiments"] = (
        df_final["label"].map(LABEL_TO_ID)
    )

    # Tên nhãn để dễ kiểm tra
    df_final["sentiment_label"] = (
        df_final["sentiments"].map(LABEL_NAMES)
    )

    # Chỉ giữ các cột cần thiết
    df_final = df_final[
        ["comments", "sentiments", "sentiment_label"]
    ]

    return df_final


# ==============================
# 6. KIỂM TRA VÀ LƯU DỮ LIỆU
# ==============================

def save_dataset(df):
    # Kiểm tra số lượng mẫu
    assert len(df) == TOTAL_SAMPLES

    # Kiểm tra phân bố nhãn
    counts = df["sentiments"].value_counts()

    assert counts.to_dict() == {
        0: 333,
        1: 333,
        2: 334
    }

    assert df["comments"].notna().all()
    assert df["sentiment_label"].notna().all()

    # Tạo thư mục lưu dữ liệu
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "uit_vsfc_1000.csv"
    excel_path = OUTPUT_DIR / "uit_vsfc_1000.xlsx"

    # Lưu CSV và Excel
    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    df.to_excel(
        excel_path,
        index=False
    )

    print("\n" + "=" * 45)
    print("HOÀN THÀNH TẠO DATASET UIT-VSFC")
    print("=" * 45)

    print(f"Tổng số mẫu: {len(df)}")

    print("\nPhân bố nhãn:")
    for label_id, label_name in LABEL_NAMES.items():
        count = int((df["sentiments"] == label_id).sum())
        percentage = count / len(df) * 100

        print(
            f"{label_name:10s}: "
            f"{count:3d} mẫu ({percentage:.1f}%)"
        )

    print("\nCác file đã lưu:")
    print(f"CSV:   {csv_path}")
    print(f"Excel: {excel_path}")

    print("\nCấu trúc dữ liệu:")
    print(df.head(5).to_string(index=False))


# ==============================
# 7. CHƯƠNG TRÌNH CHÍNH
# ==============================

def main():
    try:
        df = download_dataset()
        df = clean_data(df)
        df = sample_balanced_data(df)
        df = prepare_output(df)
        save_dataset(df)

    except Exception as error:
        print("\nĐÃ XẢY RA LỖI:")
        print(error)


if __name__ == "__main__":
    main()
