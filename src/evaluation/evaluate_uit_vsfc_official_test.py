
from pathlib import Path
import re
import html
import unicodedata

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from underthesea import word_tokenize

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================================
# 1. CẤU HÌNH
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data/alternative/official_test"
TABLE_DIR = ROOT_DIR / "results/tables/uit_vsfc/official_test"
FIGURE_DIR = ROOT_DIR / "results/figures/uit_vsfc/official_test"

MODEL_PATH = (
    ROOT_DIR
    / "models/uit_vsfc/pipeline/"
    / "pipeline_unigram_bigram_logistic_regression.joblib"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "negative": 0,
    "positive": 1,
    "neutral": 2
}

LABEL_NAMES = ["Negative", "Positive", "Neutral"]
LABEL_IDS = [0, 1, 2]


# =====================================================
# 2. KIỂM TRA MÔ HÌNH
# =====================================================

print("=" * 65)
print("ĐÁNH GIÁ PIPELINE TRÊN TEST CHÍNH THỨC - UIT-VSFC")
print("=" * 65)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Đã tải mô hình:")
print("Logistic Regression - TF-IDF Unigram + Bigram")


# =====================================================
# 3. TẢI TẬP TEST CHÍNH THỨC
# =====================================================

print("\nĐang tải tập Test chính thức UIT-VSFC...")

dataset = load_dataset(
    "ura-hcmut/UIT-VSFC",
    split="test"
)

test_df = dataset.to_pandas()

required_columns = ["text", "label"]

for column in required_columns:
    if column not in test_df.columns:
        raise ValueError(
            f"Tập Test không có cột {column}"
        )

print(f"Số mẫu Test tải được: {len(test_df)}")


# =====================================================
# 4. CHUẨN HÓA NHÃN
# =====================================================

def convert_label(value):

    if isinstance(value, str):
        label = value.strip().lower()

        if label not in LABEL_MAP:
            raise ValueError(
                f"Nhãn không hợp lệ: {value}"
            )

        return LABEL_MAP[label]

    value = int(value)

    if value not in LABEL_IDS:
        raise ValueError(
            f"Mã nhãn không hợp lệ: {value}"
        )

    return value


test_df["sentiments"] = test_df["label"].apply(
    convert_label
)


# =====================================================
# 5. TIỀN XỬ LÝ VĂN BẢN
# =====================================================

def preprocess_text(text):

    text = str(text)

    # Chuẩn hóa Unicode
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
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # Chuẩn hóa khoảng trắng
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nĐang tiền xử lý văn bản...")

test_df["clean_comments"] = (
    test_df["text"].apply(preprocess_text)
)

test_df = test_df[
    test_df["clean_comments"].str.len() > 0
].copy()


# =====================================================
# 6. TÁCH TỪ TIẾNG VIỆT
# =====================================================

print("Đang thực hiện Word Segmentation...")

test_df["segmented_comments"] = (
    test_df["clean_comments"].apply(
        lambda text: word_tokenize(
            text,
            format="text"
        )
    )
)

print("Hoàn thành tiền xử lý và tách từ.")


# =====================================================
# 7. KIỂM TRA DỮ LIỆU
# =====================================================

X_test = test_df["segmented_comments"].astype(str)
y_test = test_df["sentiments"].astype(int)

if len(test_df) == 0:
    raise ValueError("Tập Test không còn mẫu hợp lệ.")

if set(y_test.unique()) != set(LABEL_IDS):
    raise ValueError(
        "Tập Test không có đủ ba nhãn cảm xúc."
    )

print("\nPhân bố nhãn Test:")
print(
    y_test.value_counts()
    .reindex(LABEL_IDS, fill_value=0)
    .rename(index={
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    })
)


# =====================================================
# 8. DỰ ĐOÁN
# =====================================================

print("\nĐang dự đoán cảm xúc...")

y_pred = model.predict(X_test)

print("Hoàn thành dự đoán.")


# =====================================================
# 9. TÍNH TOÁN CHỈ SỐ ĐÁNH GIÁ
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

macro_precision = precision_score(
    y_test,
    y_pred,
    labels=LABEL_IDS,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    labels=LABEL_IDS,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    labels=LABEL_IDS,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    labels=LABEL_IDS,
    average="weighted",
    zero_division=0
)

correct_count = (y_test == y_pred).sum()
incorrect_count = (y_test != y_pred).sum()


# =====================================================
# 10. HIỂN THỊ KẾT QUẢ
# =====================================================

print("\n" + "=" * 65)
print("KẾT QUẢ ĐÁNH GIÁ TRÊN TEST CHÍNH THỨC")
print("=" * 65)

print(f"Tổng số mẫu: {len(y_test)}")
print(f"Dự đoán đúng: {correct_count}")
print(f"Dự đoán sai: {incorrect_count}")

print(f"\nAccuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")


# =====================================================
# 11. CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    y_test,
    y_pred,
    labels=LABEL_IDS,
    target_names=LABEL_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = (
    TABLE_DIR / "official_test_classification_report.csv"
)

report_df.to_csv(
    report_path,
    encoding="utf-8-sig"
)

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        zero_division=0
    )
)


# =====================================================
# 12. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=LABEL_IDS
)

print("\nCONFUSION MATRIX:")

print(
    pd.DataFrame(
        cm,
        index=LABEL_NAMES,
        columns=LABEL_NAMES
    )
)

fig, ax = plt.subplots(figsize=(8, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABEL_NAMES
)

display.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

ax.set_title(
    "Confusion Matrix - UIT-VSFC Official Test"
)

plt.tight_layout()

cm_path = (
    FIGURE_DIR / "official_test_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 13. LƯU CHỈ SỐ TỔNG HỢP
# =====================================================

metrics_df = pd.DataFrame([{
    "dataset": "UIT-VSFC Official Test",
    "model": "Logistic Regression",
    "configuration": "TF-IDF Unigram + Bigram",
    "test_samples": len(y_test),
    "correct_predictions": correct_count,
    "incorrect_predictions": incorrect_count,
    "accuracy": accuracy,
    "precision_macro": macro_precision,
    "recall_macro": macro_recall,
    "f1_macro": macro_f1,
    "f1_weighted": weighted_f1
}])

metrics_path = (
    TABLE_DIR / "official_test_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 14. LƯU KẾT QUẢ DỰ ĐOÁN
# =====================================================

prediction_df = pd.DataFrame({
    "comments": test_df["text"].values,
    "clean_comments": test_df["clean_comments"].values,
    "segmented_comments": test_df["segmented_comments"].values,
    "actual_label": y_test.map({
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }).values,
    "predicted_label": pd.Series(y_pred).map({
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }).values
})

prediction_df["is_correct"] = (
    prediction_df["actual_label"]
    == prediction_df["predicted_label"]
)

prediction_path = (
    TABLE_DIR / "official_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 15. LƯU TẬP TEST ĐÃ TIỀN XỬ LÝ
# =====================================================

processed_path = (
    DATA_DIR / "uit_vsfc_official_test_processed.csv"
)

test_df.to_csv(
    processed_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 16. KẾT THÚC
# =====================================================

print("\n" + "=" * 65)
print("HOÀN THÀNH ĐÁNH GIÁ TEST CHÍNH THỨC")
print("=" * 65)

print(f"- Metrics: {metrics_path}")
print(f"- Classification Report: {report_path}")
print(f"- Confusion Matrix: {cm_path}")
print(f"- Predictions: {prediction_path}")
print(f"- Processed Test: {processed_path}")

print("\nĐã hoàn thành thực nghiệm trên UIT-VSFC Official Test!")
