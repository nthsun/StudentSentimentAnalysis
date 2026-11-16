
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
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
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT_DIR / "data/alternative/processed/uit_vsfc_train.csv"
)

TEST_PATH = (
    ROOT_DIR / "data/alternative/processed/uit_vsfc_test.csv"
)

TABLE_DIR = ROOT_DIR / "results/tables/uit_vsfc"
FIGURE_DIR = ROOT_DIR / "results/figures/uit_vsfc"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_IDS = [0, 1, 2]

LABEL_NAMES = [
    "Negative",
    "Positive",
    "Neutral"
]


# =====================================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU
# =====================================================

print("=" * 55)
print("ĐÁNH GIÁ BASELINE - UIT-VSFC")
print("=" * 55)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy: {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy: {TEST_PATH}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

required_column = "sentiments"

if required_column not in train_df.columns:
    raise ValueError("Train không có cột sentiments.")

if required_column not in test_df.columns:
    raise ValueError("Test không có cột sentiments.")

y_train = train_df[required_column].astype(int)
y_test = test_df[required_column].astype(int)

if not set(y_train.unique()).issubset(set(LABEL_IDS)):
    raise ValueError("Train chứa nhãn không hợp lệ.")

if not set(y_test.unique()).issubset(set(LABEL_IDS)):
    raise ValueError("Test chứa nhãn không hợp lệ.")

print(f"Số mẫu Train: {len(y_train)}")
print(f"Số mẫu Test: {len(y_test)}")

print("\nPhân bố nhãn Train:")
print(
    y_train.value_counts()
    .reindex(LABEL_IDS, fill_value=0)
)

print("\nPhân bố nhãn Test:")
print(
    y_test.value_counts()
    .reindex(LABEL_IDS, fill_value=0)
)


# =====================================================
# 3. TẠO DỮ LIỆU ĐẦU VÀO GIẢ LẬP
# =====================================================

# DummyClassifier không cần đặc trưng văn bản.
# Sử dụng một cột giả lập độc lập với nhãn cảm xúc.

X_train_dummy = np.zeros((len(y_train), 1))
X_test_dummy = np.zeros((len(y_test), 1))


# =====================================================
# 4. HUẤN LUYỆN BASELINE
# =====================================================

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(
    X_train_dummy,
    y_train
)

y_pred = baseline.predict(
    X_test_dummy
)

predicted_label = baseline.classes_[
    np.argmax(baseline.class_prior_)
]

print("\nThông tin Baseline:")
print("Thuật toán: DummyClassifier")
print("Strategy: most_frequent")
print(f"Nhãn được dự đoán: {LABEL_NAMES[int(predicted_label)]}")


# =====================================================
# 5. TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ
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


# =====================================================
# 6. HIỂN THỊ KẾT QUẢ
# =====================================================

print("\n" + "=" * 55)
print("KẾT QUẢ ĐÁNH GIÁ BASELINE")
print("=" * 55)

print(f"Accuracy:       {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")


# =====================================================
# 7. CLASSIFICATION REPORT
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
    TABLE_DIR / "baseline_classification_report.csv"
)

report_df.to_csv(
    report_path,
    encoding="utf-8-sig"
)

print("\nBÁO CÁO PHÂN LOẠI:")
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
# 8. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=LABEL_IDS
)

cm_df = pd.DataFrame(
    cm,
    index=LABEL_NAMES,
    columns=LABEL_NAMES
)

print("\nMA TRẬN NHẦM LẪN:")
print(cm_df)

cm_path = (
    FIGURE_DIR / "confusion_matrix_baseline.png"
)

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABEL_NAMES
)

display.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

ax.set_title("Confusion Matrix - UIT-VSFC Baseline")

plt.tight_layout()
plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 9. LƯU KẾT QUẢ TỔNG HỢP
# =====================================================

results = pd.DataFrame([{
    "Model": "DummyClassifier",
    "Strategy": "most_frequent",
    "Predicted_Label": LABEL_NAMES[int(predicted_label)],
    "Accuracy": accuracy,
    "Macro_Precision": macro_precision,
    "Macro_Recall": macro_recall,
    "Macro_F1": macro_f1,
    "Weighted_F1": weighted_f1
}])

result_path = (
    TABLE_DIR / "baseline_comparison.csv"
)

results.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 10. LƯU DỰ ĐOÁN
# =====================================================

prediction_df = pd.DataFrame({
    "actual_label": y_test.map(dict(enumerate(LABEL_NAMES))),
    "predicted_label": pd.Series(y_pred).map(
        dict(enumerate(LABEL_NAMES))
    )
})

prediction_path = (
    TABLE_DIR / "baseline_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 11. KẾT THÚC
# =====================================================

print("\n" + "=" * 55)
print("ĐÃ LƯU TOÀN BỘ KẾT QUẢ")
print("=" * 55)

print(f"- {result_path}")
print(f"- {report_path}")
print(f"- {cm_path}")
print(f"- {prediction_path}")

print("\nHoàn thành đánh giá Baseline!")
