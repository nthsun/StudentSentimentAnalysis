
import sys
from pathlib import Path

import pandas as pd

# ============================================================
# 1. THIẾT LẬP ĐƯỜNG DẪN DỰ ÁN
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# Thêm thư mục gốc vào Python path để import được package src
sys.path.insert(0, str(ROOT))

from src.preprocessing.vietnamese_text_processor import (
    preprocess_and_segment
)


# ============================================================
# 2. CẤU HÌNH CÁC BỘ DỮ LIỆU
# ============================================================

DATASETS = {
    "Train (800)": (
        ROOT / "data/alternative/processed/uit_vsfc_train.csv"
    ),
    "Test (200)": (
        ROOT / "data/alternative/processed/uit_vsfc_test.csv"
    ),
    "Official Test (3166)": (
        ROOT / "data/alternative/official_test/uit_vsfc_official_test_processed.csv"
    ),
}


# ============================================================
# 3. CÁC CÂU ĐƯỢC DÙNG ĐỂ KIỂM TRA
# ============================================================

SAMPLES = [
    "Giảng viên rất nhiệt tình, giảng bài dễ hiểu và thân thiện.",
    "Tôi không hài lòng với chất lượng giảng dạy, bài tập quá khó.",
    "Nội dung môn học ở mức bình thường, không có gì đặc biệt.",
    "Môn học không quá khó nhưng cách giảng dạy chưa thực sự tốt."
]


# ============================================================
# 4. HÀM CHUẨN HÓA DỮ LIỆU
# ============================================================

def normalize_text(text):
    """
    Làm sạch và phân đoạn văn bản theo đúng quy trình
    preprocessing đã sử dụng khi huấn luyện mô hình.
    """
    if pd.isna(text):
        return ""

    return preprocess_and_segment(str(text))


def load_dataset(dataset_name, file_path):
    """
    Đọc dataset và chuẩn hóa nội dung bình luận.

    Ưu tiên cột segmented_comments nếu đã tồn tại.
    Nếu không có, chuẩn hóa từ clean_comments hoặc comments.
    """

    if not file_path.exists():
        print(f"[WARNING] Không tìm thấy file: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # Xác định cột chứa nội dung văn bản
    possible_columns = [
        "segmented_comments",
        "clean_comments",
        "comments"
    ]

    text_column = next(
        (col for col in possible_columns if col in df.columns),
        None
    )

    if text_column is None:
        print(
            f"[WARNING] Dataset {dataset_name} "
            "không có cột văn bản phù hợp."
        )
        return None

    # Chuẩn hóa toàn bộ nội dung để so sánh cùng một định dạng
    df["normalized_text"] = df[text_column].apply(normalize_text)

    return df


# ============================================================
# 5. CHUẨN BỊ CÁC CÂU KIỂM TRA
# ============================================================

normalized_samples = {
    sample: normalize_text(sample)
    for sample in SAMPLES
}


# ============================================================
# 6. KIỂM TRA TRÙNG LẶP CHÍNH XÁC
# ============================================================

def check_overlap():

    print("\n" + "=" * 75)
    print("KIỂM TRA TRÙNG LẶP CÂU DỰ ĐOÁN VỚI CÁC BỘ DỮ LIỆU")
    print("=" * 75)

    datasets_loaded = {}

    # Đọc các bộ dữ liệu
    for dataset_name, file_path in DATASETS.items():

        df = load_dataset(dataset_name, file_path)

        if df is not None:
            datasets_loaded[dataset_name] = df

            print(
                f"\n[OK] {dataset_name}: "
                f"{len(df)} mẫu"
            )

    if not datasets_loaded:
        print("\nKhông có dataset nào được tải thành công.")
        return

    total_matches = 0

    # Kiểm tra từng câu
    for index, (original_text, normalized_text) in enumerate(
        normalized_samples.items(),
        start=1
    ):

        print("\n" + "-" * 75)
        print(f"CÂU {index}")
        print(f"Văn bản gốc: {original_text}")
        print(f"Văn bản sau xử lý: {normalized_text}")

        found = False

        for dataset_name, df in datasets_loaded.items():

            matches = df[
                df["normalized_text"] == normalized_text
            ]

            if not matches.empty:

                found = True
                total_matches += len(matches)

                print(
                    f"\n[TRÙNG] Tìm thấy trong: {dataset_name}"
                )

                # Hiển thị nhãn thực tế nếu có
                label_columns = [
                    "sentiments",
                    "sentiment_label",
                    "actual_label"
                ]

                label_column = next(
                    (
                        col for col in label_columns
                        if col in matches.columns
                    ),
                    None
                )

                if label_column:
                    print(
                        "Nhãn trong dataset:",
                        matches[label_column].tolist()
                    )

                print(
                    "Số lần xuất hiện:",
                    len(matches)
                )

        if not found:
            print(
                "\n[KHÔNG TRÙNG] "
                "Không tìm thấy câu giống hệt trong các dataset."
            )

    # Tổng kết
    print("\n" + "=" * 75)
    print("TỔNG KẾT")
    print("=" * 75)

    print(f"Tổng số câu kiểm tra: {len(SAMPLES)}")
    print(f"Tổng số kết quả trùng: {total_matches}")

    if total_matches == 0:
        print(
            "\nKết luận: Không phát hiện câu nào trùng khớp "
            "chính xác sau tiền xử lý."
        )
    else:
        print(
            "\nKết luận: Có câu trùng khớp chính xác với "
            "ít nhất một bộ dữ liệu."
        )

    print(
        "\nLưu ý: Kiểm tra này chỉ phát hiện trùng lặp chính xác "
        "sau tiền xử lý, không phát hiện các câu có nội dung "
        "tương tự nhưng cách diễn đạt khác nhau."
    )


# ============================================================
# 7. CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":
    check_overlap()
