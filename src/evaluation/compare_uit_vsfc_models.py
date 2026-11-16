
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_RESULT_PATH = (
    ROOT_DIR / "results/tables/uit_vsfc/model_comparison.csv"
)

BASELINE_RESULT_PATH = (
    ROOT_DIR / "results/tables/uit_vsfc/baseline_comparison.csv"
)

TABLE_DIR = ROOT_DIR / "results/tables/uit_vsfc"
FIGURE_DIR = ROOT_DIR / "results/figures/uit_vsfc"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. KIỂM TRA VÀ ĐỌC DỮ LIỆU
# =====================================================

print("=" * 65)
print("TỔNG HỢP VÀ SO SÁNH CÁC MÔ HÌNH - UIT-VSFC")
print("=" * 65)

if not MODEL_RESULT_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file kết quả mô hình: {MODEL_RESULT_PATH}"
    )

if not BASELINE_RESULT_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file Baseline: {BASELINE_RESULT_PATH}"
    )

model_df = pd.read_csv(MODEL_RESULT_PATH)
baseline_df = pd.read_csv(BASELINE_RESULT_PATH)

print(f"Số kết quả mô hình ML: {len(model_df)}")
print(f"Số kết quả Baseline: {len(baseline_df)}")


# =====================================================
# 3. CHUẨN HÓA DỮ LIỆU MÔ HÌNH
# =====================================================

required_model_columns = [
    "configuration",
    "model",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "cv_macro_f1",
    "best_params"
]

missing_columns = [
    col for col in required_model_columns
    if col not in model_df.columns
]

if missing_columns:
    raise ValueError(
        f"Thiếu cột trong model_comparison.csv: {missing_columns}"
    )

model_results = model_df[required_model_columns].copy()

model_results = model_results.rename(columns={
    "configuration": "Configuration",
    "model": "Model",
    "accuracy": "Accuracy",
    "precision_macro": "Macro_Precision",
    "recall_macro": "Macro_Recall",
    "f1_macro": "Macro_F1",
    "f1_weighted": "Weighted_F1",
    "cv_macro_f1": "CV_Macro_F1",
    "best_params": "Best_Params"
})

model_results["Configuration"] = (
    model_results["Configuration"].replace({
        "unigram": "Unigram",
        "unigram_bigram": "Unigram + Bigram"
    })
)


# =====================================================
# 4. CHUẨN HÓA DỮ LIỆU BASELINE
# =====================================================

required_baseline_columns = [
    "Model",
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1"
]

missing_columns = [
    col for col in required_baseline_columns
    if col not in baseline_df.columns
]

if missing_columns:
    raise ValueError(
        f"Thiếu cột trong baseline_comparison.csv: {missing_columns}"
    )

baseline_results = baseline_df[required_baseline_columns].copy()

baseline_results = baseline_results.rename(columns={
    "Model": "Model"
})

baseline_results["Configuration"] = "Không sử dụng TF-IDF"

baseline_results["CV_Macro_F1"] = float("nan")
baseline_results["Best_Params"] = baseline_df.get(
    "Strategy",
    pd.Series(["most_frequent"] * len(baseline_df))
)

baseline_results = baseline_results[
    [
        "Configuration",
        "Model",
        "Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_F1",
        "CV_Macro_F1",
        "Best_Params"
    ]
]


# =====================================================
# 5. GHÉP KẾT QUẢ 7 MÔ HÌNH
# =====================================================

comparison_df = pd.concat(
    [baseline_results, model_results],
    ignore_index=True
)

if len(comparison_df) != 7:
    raise ValueError(
        f"Cần có đúng 7 kết quả, nhưng hiện có {len(comparison_df)}."
    )

# Sắp xếp theo Macro F1 giảm dần
comparison_df = comparison_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)

# Thêm thứ hạng
comparison_df.insert(
    0,
    "Rank",
    range(1, len(comparison_df) + 1)
)


# =====================================================
# 6. HIỂN THỊ BẢNG SO SÁNH
# =====================================================

print("\n" + "=" * 65)
print("BẢNG SO SÁNH 7 MÔ HÌNH")
print("=" * 65)

display_columns = [
    "Rank",
    "Configuration",
    "Model",
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
    "CV_Macro_F1"
]

print(
    comparison_df[display_columns].to_string(
        index=False,
        formatters={
            "Accuracy": lambda x: f"{x:.4f}",
            "Macro_Precision": lambda x: f"{x:.4f}",
            "Macro_Recall": lambda x: f"{x:.4f}",
            "Macro_F1": lambda x: f"{x:.4f}",
            "Weighted_F1": lambda x: f"{x:.4f}",
            "CV_Macro_F1": lambda x: (
                f"{x:.4f}" if pd.notna(x) else "N/A"
            )
        }
    )
)


# =====================================================
# 7. XÁC ĐỊNH MÔ HÌNH TỐT NHẤT
# =====================================================

best_model = comparison_df.iloc[0]

print("\n" + "=" * 65)
print("MÔ HÌNH CÓ MACRO F1 CAO NHẤT")
print("=" * 65)

print(f"Model: {best_model['Model']}")
print(f"Configuration: {best_model['Configuration']}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")
print(f"Macro F1: {best_model['Macro_F1']:.4f}")


# =====================================================
# 8. TÍNH MỨC CẢI THIỆN SO VỚI BASELINE
# =====================================================

baseline_f1 = baseline_results["Macro_F1"].iloc[0]
baseline_accuracy = baseline_results["Accuracy"].iloc[0]

comparison_df["Accuracy_Improvement_vs_Baseline"] = (
    comparison_df["Accuracy"] - baseline_accuracy
)

comparison_df["Macro_F1_Improvement_vs_Baseline"] = (
    comparison_df["Macro_F1"] - baseline_f1
)


# =====================================================
# 9. LƯU BẢNG SO SÁNH
# =====================================================

output_path = (
    TABLE_DIR / "uit_vsfc_all_models_comparison.csv"
)

comparison_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nĐã lưu bảng so sánh: {output_path}")


# =====================================================
# 10. BIỂU ĐỒ SO SÁNH ACCURACY
# =====================================================

plot_df = comparison_df.copy()

plot_df["Model_Label"] = (
    plot_df["Model"] + " - " + plot_df["Configuration"]
)

fig, ax = plt.subplots(figsize=(12, 7))

ax.barh(
    plot_df["Model_Label"],
    plot_df["Accuracy"]
)

ax.set_xlabel("Accuracy")
ax.set_ylabel("Model")
ax.set_title("Accuracy Comparison - UIT-VSFC")

ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.4)

for index, value in enumerate(plot_df["Accuracy"]):
    ax.text(
        value + 0.01,
        index,
        f"{value:.2%}",
        va="center"
    )

plt.tight_layout()

accuracy_figure = (
    FIGURE_DIR / "uit_vsfc_accuracy_comparison.png"
)

plt.savefig(
    accuracy_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 11. BIỂU ĐỒ SO SÁNH MACRO F1
# =====================================================

fig, ax = plt.subplots(figsize=(12, 7))

ax.barh(
    plot_df["Model_Label"],
    plot_df["Macro_F1"]
)

ax.set_xlabel("Macro F1")
ax.set_ylabel("Model")
ax.set_title("Macro F1 Comparison - UIT-VSFC")

ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.4)

for index, value in enumerate(plot_df["Macro_F1"]):
    ax.text(
        value + 0.01,
        index,
        f"{value:.2%}",
        va="center"
    )

plt.tight_layout()

f1_figure = (
    FIGURE_DIR / "uit_vsfc_macro_f1_comparison.png"
)

plt.savefig(
    f1_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 12. KẾT THÚC
# =====================================================

print("\n" + "=" * 65)
print("HOÀN THÀNH TỔNG HỢP KẾT QUẢ")
print("=" * 65)

print(f"- Bảng so sánh: {output_path}")
print(f"- Biểu đồ Accuracy: {accuracy_figure}")
print(f"- Biểu đồ Macro F1: {f1_figure}")

print("\nĐã tổng hợp thành công Baseline và 6 mô hình ML.")
