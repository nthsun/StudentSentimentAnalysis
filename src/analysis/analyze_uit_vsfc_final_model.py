
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score
)

# ============================================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "results" / "tables" / "uit_vsfc"
    / "official_test" / "official_test_predictions.csv"
)

TABLE_DIR = (
    ROOT / "results" / "tables" / "uit_vsfc"
    / "final_model_analysis"
)

FIGURE_DIR = (
    ROOT / "results" / "figures" / "uit_vsfc"
    / "final_model_analysis"
)

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Positive", "Neutral"]

# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("PHÂN TÍCH MÔ HÌNH LOGISTIC REGRESSION - UIT-VSFC")
print("=" * 70)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file dự đoán: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "comments",
    "actual_label",
    "predicted_label",
    "is_correct"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Thiếu các cột bắt buộc: {missing_columns}"
    )

df = df.dropna(
    subset=["actual_label", "predicted_label"]
).copy()

df["actual_label"] = df["actual_label"].astype(str)
df["predicted_label"] = df["predicted_label"].astype(str)

# Kiểm tra nhãn
unknown_labels = (
    set(df["actual_label"].unique())
    | set(df["predicted_label"].unique())
) - set(LABELS)

if unknown_labels:
    raise ValueError(
        f"Phát hiện nhãn không hợp lệ: {unknown_labels}"
    )

y_true = df["actual_label"]
y_pred = df["predicted_label"]

print(f"\nTổng số mẫu: {len(df)}")

print("\nPhân bố nhãn thực tế:")
print(y_true.value_counts().reindex(LABELS, fill_value=0))

print("\nPhân bố nhãn dự đoán:")
print(y_pred.value_counts().reindex(LABELS, fill_value=0))


# ============================================================
# 3. TÍNH CÁC CHỈ SỐ TỔNG QUAN
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0
    )
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=LABELS,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=LABELS,
    average="weighted",
    zero_division=0
)

metrics_df = pd.DataFrame([{
    "test_samples": len(df),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "correct_predictions": int((y_true == y_pred).sum()),
    "incorrect_predictions": int((y_true != y_pred).sum())
}])

metrics_df.to_csv(
    TABLE_DIR / "overall_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("CHỈ SỐ ĐÁNH GIÁ TỔNG QUAN")
print("=" * 70)

print(f"Accuracy:     {accuracy:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")
print(f"Số dự đoán đúng: {(y_true == y_pred).sum()}")
print(f"Số dự đoán sai:  {(y_true != y_pred).sum()}")


# ============================================================
# 4. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=LABELS,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    TABLE_DIR / "classification_report.csv",
    encoding="utf-8-sig"
)

class_metrics = pd.DataFrame({
    "sentiment": LABELS,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "support": support
})

class_metrics.to_csv(
    TABLE_DIR / "class_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("KẾT QUẢ THEO TỪNG LỚP CẢM XÚC")
print("=" * 70)

print(
    class_metrics.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1_score": "{:.4f}".format
        }
    )
)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=LABELS
)

cm_df = pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
)

cm_df.index.name = "Actual"
cm_df.columns.name = "Predicted"

cm_df.to_csv(
    TABLE_DIR / "confusion_matrix.csv",
    encoding="utf-8-sig"
)

# Ma trận số lượng
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABELS,
    yticklabels=LABELS
)

plt.title("Confusion Matrix - UIT-VSFC Official Test")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. CONFUSION MATRIX THEO TỶ LỆ
# ============================================================

row_sums = cm.sum(axis=1, keepdims=True)

cm_normalized = (
    cm / row_sums
)

cm_normalized_df = pd.DataFrame(
    cm_normalized,
    index=LABELS,
    columns=LABELS
)

cm_normalized_df.to_csv(
    TABLE_DIR / "confusion_matrix_normalized.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_normalized_df,
    annot=True,
    fmt=".2%",
    cmap="YlGnBu",
    xticklabels=LABELS,
    yticklabels=LABELS
)

plt.title("Normalized Confusion Matrix - UIT-VSFC")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "confusion_matrix_normalized.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. BIỂU ĐỒ SO SÁNH PRECISION, RECALL, F1
# ============================================================

plot_df = class_metrics.melt(
    id_vars="sentiment",
    value_vars=["precision", "recall", "f1_score"],
    var_name="metric",
    value_name="score"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=plot_df,
    x="sentiment",
    y="score",
    hue="metric"
)

plt.title("Precision, Recall and F1-score by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "class_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. PHÂN TÍCH CÁC CẶP NHÃN BỊ NHẦM LẪN
# ============================================================

errors = df[
    df["actual_label"] != df["predicted_label"]
].copy()

errors["error_type"] = (
    errors["actual_label"]
    + " -> "
    + errors["predicted_label"]
)

error_summary = (
    errors.groupby("error_type")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

error_summary["percentage_of_errors"] = (
    error_summary["count"] / max(len(errors), 1) * 100
)

error_summary.to_csv(
    TABLE_DIR / "error_type_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("THỐNG KÊ CÁC CẶP NHÃN BỊ NHẦM LẪN")
print("=" * 70)

print(error_summary.to_string(index=False))


# ============================================================
# 9. LƯU TOÀN BỘ CÁC MẪU DỰ ĐOÁN SAI
# ============================================================

error_columns = [
    col for col in [
        "comments",
        "clean_comments",
        "segmented_comments",
        "actual_label",
        "predicted_label",
        "error_type"
    ]
    if col in errors.columns
]

errors[error_columns].to_csv(
    TABLE_DIR / "misclassified_samples.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. THỐNG KÊ CÁC TRƯỜNG HỢP DỰ ĐOÁN SAI THEO NHÃN THỰC TẾ
# ============================================================

actual_error_summary = (
    errors.groupby("actual_label")
    .size()
    .reindex(LABELS, fill_value=0)
    .reset_index(name="incorrect_count")
)

actual_error_summary.rename(
    columns={"actual_label": "sentiment"},
    inplace=True
)

actual_error_summary.to_csv(
    TABLE_DIR / "errors_by_actual_label.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. HIỂN THỊ MỘT SỐ MẪU DỰ ĐOÁN SAI
# ============================================================

print("\n" + "=" * 70)
print("10 MẪU DỰ ĐOÁN SAI ĐẦU TIÊN")
print("=" * 70)

if len(errors) > 0:
    print(
        errors[error_columns]
        .head(10)
        .to_string(index=False)
    )
else:
    print("Không có mẫu dự đoán sai.")


# ============================================================
# 12. KẾT THÚC
# ============================================================

print("\n" + "=" * 70)
print("HOÀN THÀNH PHÂN TÍCH MÔ HÌNH")
print("=" * 70)

print(f"\nBảng kết quả: {TABLE_DIR}")
print(f"Biểu đồ:      {FIGURE_DIR}")

print("\nCác file đã tạo:")
for path in sorted(TABLE_DIR.glob("*.csv")):
    print(f"- {path.name}")

for path in sorted(FIGURE_DIR.glob("*.png")):
    print(f"- {path.name}")

print("\nLưu ý:")
print(
    "Phân tích được thực hiện từ các dự đoán đã lưu. "
    "Không huấn luyện lại hoặc điều chỉnh mô hình."
)
