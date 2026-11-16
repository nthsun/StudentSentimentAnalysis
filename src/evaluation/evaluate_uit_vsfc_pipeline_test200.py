
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

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

TEST_PATH = (
    ROOT_DIR
    / "data/alternative/processed/uit_vsfc_test.csv"
)

MODEL_DIR = (
    ROOT_DIR / "models/uit_vsfc/pipeline"
)

RESULT_DIR = (
    ROOT_DIR / "results/tables/uit_vsfc/pipeline_test200"
)

FIGURE_DIR = (
    ROOT_DIR / "results/figures/uit_vsfc/pipeline_test200"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_IDS = [0, 1, 2]

LABEL_NAMES = [
    "Negative",
    "Positive",
    "Neutral"
]


# =====================================================
# 2. KIỂM TRA VÀ ĐỌC TEST
# =====================================================

print("=" * 65)
print("ĐÁNH GIÁ 6 PIPELINE TRÊN TEST 200 MẪU")
print("=" * 65)

if not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu Test: {TEST_PATH}"
    )

test_df = pd.read_csv(TEST_PATH)

required_columns = [
    "segmented_comments",
    "sentiments"
]

for column in required_columns:
    if column not in test_df.columns:
        raise ValueError(
            f"Thiếu cột {column} trong Test."
        )

test_df = test_df.dropna(
    subset=required_columns
).copy()

X_test = test_df["segmented_comments"].astype(str)
y_test = test_df["sentiments"].astype(int)

if len(test_df) == 0:
    raise ValueError("Tập Test không có dữ liệu.")

if not set(y_test.unique()).issubset(set(LABEL_IDS)):
    raise ValueError("Tập Test chứa nhãn không hợp lệ.")

print(f"Số mẫu Test: {len(test_df)}")

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
# 3. KHAI BÁO 6 PIPELINE
# =====================================================

model_configs = [
    {
        "configuration": "Unigram",
        "model": "Naive Bayes",
        "filename": "pipeline_unigram_naive_bayes.joblib"
    },
    {
        "configuration": "Unigram",
        "model": "Logistic Regression",
        "filename": "pipeline_unigram_logistic_regression.joblib"
    },
    {
        "configuration": "Unigram",
        "model": "Linear SVM",
        "filename": "pipeline_unigram_linear_svm.joblib"
    },
    {
        "configuration": "Unigram + Bigram",
        "model": "Naive Bayes",
        "filename": "pipeline_unigram_bigram_naive_bayes.joblib"
    },
    {
        "configuration": "Unigram + Bigram",
        "model": "Logistic Regression",
        "filename": "pipeline_unigram_bigram_logistic_regression.joblib"
    },
    {
        "configuration": "Unigram + Bigram",
        "model": "Linear SVM",
        "filename": "pipeline_unigram_bigram_linear_svm.joblib"
    }
]


# =====================================================
# 4. ĐÁNH GIÁ CÁC PIPELINE
# =====================================================

all_results = []
all_reports = []

prediction_df = pd.DataFrame({
    "segmented_comments": X_test.values,
    "actual_label": y_test.map({
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }).values
})

for index, config in enumerate(model_configs, start=1):

    print("\n" + "=" * 65)
    print(f"ĐÁNH GIÁ MÔ HÌNH {index}/6")
    print(f"Model: {config['model']}")
    print(f"Configuration: {config['configuration']}")
    print("=" * 65)

    model_path = MODEL_DIR / config["filename"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {model_path}"
        )

    model = joblib.load(model_path)

    # Dự đoán trực tiếp bằng Pipeline
    y_pred = model.predict(X_test)

    # Các chỉ số đánh giá
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        labels=LABEL_IDS,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
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

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    # Lưu kết quả
    all_results.append({
        "configuration": config["configuration"],
        "model": config["model"],
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1
    })

    # Classification Report
    report = classification_report(
        y_test,
        y_pred,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.insert(
        0,
        "model",
        config["model"]
    )

    report_df.insert(
        0,
        "configuration",
        config["configuration"]
    )

    all_reports.append(report_df)

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=LABEL_IDS
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

    ax.set_title(
        f"{config['model']} - {config['configuration']}"
    )

    plt.tight_layout()

    figure_name = (
        f"confusion_matrix_"
        f"{config['configuration'].lower().replace(' ', '_').replace('+', 'plus')}_"
        f"{config['model'].lower().replace(' ', '_')}.png"
    )

    plt.savefig(
        FIGURE_DIR / figure_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Lưu dự đoán từng mô hình
    prediction_column = (
        f"predicted_{config['configuration'].lower().replace(' ', '_').replace('+', 'plus')}_"
        f"{config['model'].lower().replace(' ', '_')}"
    )

    prediction_df[prediction_column] = pd.Series(
        y_pred,
        index=prediction_df.index
    ).map({
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    })


# =====================================================
# 5. TỔNG HỢP KẾT QUẢ
# =====================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by="f1_macro",
    ascending=False
).reset_index(drop=True)

results_df.insert(
    0,
    "rank",
    range(1, len(results_df) + 1)
)

print("\n" + "=" * 75)
print("BẢNG SO SÁNH 6 PIPELINE TRÊN TEST 200")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        formatters={
            "accuracy": lambda x: f"{x:.4f}",
            "precision_macro": lambda x: f"{x:.4f}",
            "recall_macro": lambda x: f"{x:.4f}",
            "f1_macro": lambda x: f"{x:.4f}",
            "f1_weighted": lambda x: f"{x:.4f}"
        }
    )
)


# =====================================================
# 6. LƯU BẢNG SO SÁNH
# =====================================================

results_path = (
    RESULT_DIR / "pipeline_test200_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 7. LƯU CLASSIFICATION REPORT
# =====================================================

reports_path = (
    RESULT_DIR / "pipeline_test200_classification_reports.csv"
)

pd.concat(
    all_reports,
    ignore_index=True
).to_csv(
    reports_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 8. LƯU DỰ ĐOÁN
# =====================================================

prediction_path = (
    RESULT_DIR / "pipeline_test200_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 9. BIỂU ĐỒ SO SÁNH MACRO F1
# =====================================================

plot_df = results_df.copy()

plot_df["model_label"] = (
    plot_df["model"] + " - " + plot_df["configuration"]
)

fig, ax = plt.subplots(figsize=(11, 6))

ax.barh(
    plot_df["model_label"],
    plot_df["f1_macro"]
)

ax.set_xlabel("Macro F1")
ax.set_title("Pipeline Comparison - Test 200")
ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.4)

for i, value in enumerate(plot_df["f1_macro"]):
    ax.text(
        value + 0.01,
        i,
        f"{value:.2%}",
        va="center"
    )

plt.tight_layout()

comparison_figure = (
    FIGURE_DIR / "pipeline_test200_macro_f1.png"
)

plt.savefig(
    comparison_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 10. MÔ HÌNH CÓ KẾT QUẢ CAO NHẤT
# =====================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 65)
print("MÔ HÌNH CÓ MACRO F1 CAO NHẤT")
print("=" * 65)

print(f"Model: {best_model['model']}")
print(f"Configuration: {best_model['configuration']}")
print(f"Accuracy: {best_model['accuracy']:.4f}")
print(f"Macro F1: {best_model['f1_macro']:.4f}")


# =====================================================
# 11. KẾT THÚC
# =====================================================

print("\n" + "=" * 65)
print("HOÀN THÀNH ĐÁNH GIÁ PIPELINE")
print("=" * 65)

print(f"- Bảng so sánh: {results_path}")
print(f"- Classification Report: {reports_path}")
print(f"- Dự đoán: {prediction_path}")
print(f"- Biểu đồ: {comparison_figure}")
print(f"- Confusion Matrix: {FIGURE_DIR}")

print("\nĐã đánh giá thành công 6 Pipeline trên Test 200!")
