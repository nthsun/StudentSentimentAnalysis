from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "alternative" / "processed"
METRIC_DIR = ROOT / "results" / "metrics" / "uit_vsfc"
FIGURE_DIR = ROOT / "results" / "figures" / "uit_vsfc"
TABLE_DIR = ROOT / "results" / "tables" / "uit_vsfc"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [0, 1, 2]

LABEL_NAMES = [
    "Negative",
    "Positive",
    "Neutral"
]

PREDICTION_FILE = (
    METRIC_DIR /
    "unigram_logistic_regression_predictions.csv"
)

TEST_FILE = DATA_DIR / "uit_vsfc_test.csv"


def main():

    predictions = pd.read_csv(PREDICTION_FILE)
    test_data = pd.read_csv(TEST_FILE)

    # Ghép nội dung với kết quả dự đoán
    results = test_data[
        ["segmented_comments", "sentiments"]
    ].copy()

    results = results.rename(
        columns={"sentiments": "actual"}
    )

    results["predicted"] = predictions["predicted"].values

    results["actual_label"] = (
        results["actual"].map(dict(enumerate(LABEL_NAMES)))
    )

    results["predicted_label"] = (
        results["predicted"].map(dict(enumerate(LABEL_NAMES)))
    )

    # Lọc các dự đoán sai
    errors = results[
        results["actual"] != results["predicted"]
    ].copy()

    errors.to_csv(
        TABLE_DIR / "misclassified_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Ma trận nhầm lẫn
    cm = confusion_matrix(
        results["actual"],
        results["predicted"],
        labels=LABELS
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )

    plt.title("Confusion Matrix - Logistic Regression Unigram")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "confusion_matrix_best_model.png",
        dpi=300
    )

    plt.close()

    # Báo cáo từng lớp
    report = classification_report(
        results["actual"],
        results["predicted"],
        labels=LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        TABLE_DIR / "best_model_classification_report.csv",
        encoding="utf-8-sig"
    )

    print("=" * 55)
    print("PHÂN TÍCH LỖI MÔ HÌNH")
    print("=" * 55)

    print("Tổng số mẫu:", len(results))
    print("Số dự đoán đúng:", len(results) - len(errors))
    print("Số dự đoán sai:", len(errors))

    print("\nMa trận nhầm lẫn:")
    print(
        pd.DataFrame(
            cm,
            index=LABEL_NAMES,
            columns=LABEL_NAMES
        )
    )

    print("\nPhân bố lỗi theo nhãn thực tế:")
    print(errors["actual_label"].value_counts())

    print("\n10 mẫu dự đoán sai đầu tiên:")
    print(
        errors[
            [
                "segmented_comments",
                "actual_label",
                "predicted_label"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nĐã lưu kết quả tại:", TABLE_DIR)


if __name__ == "__main__":
    main()