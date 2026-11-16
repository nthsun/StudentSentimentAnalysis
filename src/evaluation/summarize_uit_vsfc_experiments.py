
from pathlib import Path
import pandas as pd

# ============================================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TABLE_DIR = ROOT / "results" / "tables" / "uit_vsfc"
OUTPUT_DIR = TABLE_DIR / "final_summary"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Initial ML models": TABLE_DIR / "model_comparison.csv",

    "Baseline": TABLE_DIR / "baseline_comparison.csv",

    "Pipeline Cross-validation":
        TABLE_DIR / "pipeline" / "pipeline_cv_comparison.csv",

    "Pipeline Test 200":
        TABLE_DIR / "pipeline_test200" / "pipeline_test200_comparison.csv",

    "Official Test 3166":
        TABLE_DIR / "official_test" / "official_test_metrics.csv",
}

# ============================================================
# 2. HÀM CHUẨN HÓA TÊN CỘT
# ============================================================

COLUMN_ALIASES = {
    "accuracy": [
        "accuracy", "Accuracy"
    ],
    "precision_macro": [
        "precision_macro",
        "macro_precision",
        "Macro Precision",
        "Macro_Precision"
    ],
    "recall_macro": [
        "recall_macro",
        "macro_recall",
        "Macro Recall",
        "Macro_Recall"
    ],
    "f1_macro": [
        "f1_macro",
        "macro_f1",
        "Macro F1",
        "Macro_F1"
    ],
    "f1_weighted": [
        "f1_weighted",
        "weighted_f1",
        "Weighted F1",
        "Weighted_F1"
    ],
    "cv_f1_macro": [
        "cv_f1_macro",
        "cv_macro_f1",
        "best_cv_f1",
        "mean_cv_f1"
    ],
}

def find_column(df, aliases):
    for col in aliases:
        if col in df.columns:
            return col
    return None


def standardize_table(df, evaluation_name):
    result = pd.DataFrame()

    # Giữ thông tin tên mô hình và cấu hình
    model_col = find_column(
        df, ["model", "Model", "classifier"]
    )

    config_col = find_column(
        df, ["configuration", "Configuration", "feature"]
    )

    result["evaluation"] = [evaluation_name] * len(df)

    result["model"] = (
        df[model_col].astype(str).values
        if model_col else "Unknown"
    )

    result["configuration"] = (
        df[config_col].astype(str).values
        if config_col else "Not specified"
    )

    # Chuẩn hóa các chỉ số
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "cv_f1_macro",
    ]

    for metric in metric_names:
        source_col = find_column(
            df, COLUMN_ALIASES.get(metric, [])
        )

        result[metric] = (
            pd.to_numeric(df[source_col], errors="coerce").values
            if source_col else float("nan")
        )

    return result


# ============================================================
# 3. ĐỌC VÀ TỔNG HỢP CÁC KẾT QUẢ
# ============================================================

all_results = []
missing_files = []

print("=" * 70)
print("TỔNG HỢP KẾT QUẢ THỰC NGHIỆM UIT-VSFC")
print("=" * 70)

for evaluation_name, file_path in FILES.items():

    print(f"\nĐang kiểm tra: {evaluation_name}")

    if not file_path.exists():
        print(f"KHÔNG TÌM THẤY: {file_path}")
        missing_files.append(str(file_path))
        continue

    try:
        df = pd.read_csv(file_path)

        print(f"Số dòng: {len(df)}")
        print(f"Các cột: {list(df.columns)}")

        standardized = standardize_table(
            df, evaluation_name
        )

        all_results.append(standardized)

        print("Đọc thành công.")

    except Exception as error:
        print(f"Lỗi khi đọc file: {error}")


if not all_results:
    raise RuntimeError(
        "Không đọc được file kết quả nào. "
        "Hãy kiểm tra lại đường dẫn."
    )


# ============================================================
# 4. GHÉP BẢNG KẾT QUẢ
# ============================================================

summary = pd.concat(
    all_results,
    ignore_index=True
)

# Sắp xếp theo nhóm đánh giá và Macro F1
summary = summary.sort_values(
    by=["evaluation", "f1_macro"],
    ascending=[True, False],
    na_position="last"
)

summary = summary.round(4)


# ============================================================
# 5. XUẤT BẢNG TỔNG HỢP
# ============================================================

summary_path = (
    OUTPUT_DIR / "uit_vsfc_complete_experiment_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. XUẤT RIÊNG BẢNG PIPELINE CROSS-VALIDATION
# ============================================================

cv_summary = summary[
    summary["evaluation"] == "Pipeline Cross-validation"
].copy()

if not cv_summary.empty:
    cv_summary = cv_summary.sort_values(
        "cv_f1_macro",
        ascending=False,
        na_position="last"
    )

    cv_summary.to_csv(
        OUTPUT_DIR / "uit_vsfc_pipeline_cv_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 7. XUẤT RIÊNG BẢNG TEST 200
# ============================================================

test200_summary = summary[
    summary["evaluation"] == "Pipeline Test 200"
].copy()

if not test200_summary.empty:
    test200_summary = test200_summary.sort_values(
        "f1_macro",
        ascending=False
    )

    test200_summary.to_csv(
        OUTPUT_DIR / "uit_vsfc_pipeline_test200_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 8. HIỂN THỊ KẾT QUẢ
# ============================================================

print("\n" + "=" * 100)
print("BẢNG TỔNG HỢP KẾT QUẢ")
print("=" * 100)

print(
    summary.to_string(
        index=False,
        na_rep="N/A"
    )
)

print("\n" + "=" * 70)
print("HOÀN THÀNH")
print("=" * 70)

print(f"\nBảng tổng hợp: {summary_path}")

print(f"\nThư mục kết quả: {OUTPUT_DIR}")

if missing_files:
    print("\nCẢNH BÁO: Một số file chưa được tìm thấy:")
    for path in missing_files:
        print(f"- {path}")

print(
    "\nLưu ý: Các tập đánh giá khác nhau không được "
    "so sánh trực tiếp như cùng một tập Test."
)
