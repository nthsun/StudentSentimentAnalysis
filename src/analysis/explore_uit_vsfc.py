
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# 1. CẤU HÌNH
# =====================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "alternative" / "uit_vsfc_1000.csv"

FIGURE_DIR = ROOT / "results" / "figures" / "uit_vsfc"
TABLE_DIR = ROOT / "results" / "tables" / "uit_vsfc"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}


# =====================================
# 2. ĐỌC DỮ LIỆU
# =====================================

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "comments",
    "sentiments",
    "sentiment_label"
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Thiếu cột dữ liệu. Các cột hiện tại: {df.columns.tolist()}"
    )

print("=" * 60)
print("KHÁM PHÁ DATASET UIT-VSFC")
print("=" * 60)

print("\n1. KÍCH THƯỚC DỮ LIỆU")
print(f"Số dòng: {df.shape[0]}")
print(f"Số cột: {df.shape[1]}")

print("\n2. THÔNG TIN CÁC CỘT")
print(df.info())

print("\n3. 5 DÒNG DỮ LIỆU ĐẦU TIÊN")
print(df.head().to_string(index=False))

print("\n4. THỐNG KÊ DỮ LIỆU")
print(df.describe(include="all").to_string())

# =====================================
# 3. KIỂM TRA DỮ LIỆU THIẾU
# =====================================

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": df.isnull().mean() * 100
})

print("\n5. DỮ LIỆU THIẾU")
print(missing)

missing.to_csv(
    TABLE_DIR / "missing_values.csv",
    encoding="utf-8-sig"
)

# =====================================
# 4. KIỂM TRA DỮ LIỆU TRÙNG LẶP
# =====================================

duplicate_text = df["comments"].duplicated().sum()
duplicate_rows = df.duplicated().sum()

print("\n6. DỮ LIỆU TRÙNG LẶP")
print(f"Số bình luận trùng: {duplicate_text}")
print(f"Số dòng trùng hoàn toàn: {duplicate_rows}")

# Kiểm tra một bình luận có nhiều nhãn khác nhau không
label_conflicts = (
    df.groupby("comments")["sentiments"]
    .nunique()
)

conflict_count = int((label_conflicts > 1).sum())

print(
    "Số bình luận có nhãn cảm xúc mâu thuẫn:",
    conflict_count
)

# =====================================
# 5. PHÂN BỐ NHÃN
# =====================================

label_counts = (
    df["sentiments"]
    .value_counts()
    .reindex(LABEL_NAMES.keys(), fill_value=0)
)

label_distribution = pd.DataFrame({
    "label_id": label_counts.index,
    "label_name": [
        LABEL_NAMES[i] for i in label_counts.index
    ],
    "count": label_counts.values,
    "percentage": (
        label_counts.values / len(df) * 100
    ).round(2)
})

print("\n7. PHÂN BỐ NHÃN")
print(label_distribution.to_string(index=False))

label_distribution.to_csv(
    TABLE_DIR / "label_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    label_distribution["label_name"],
    label_distribution["count"]
)

plt.title("Phân bố cảm xúc trong UIT-VSFC")
plt.xlabel("Nhãn cảm xúc")
plt.ylabel("Số lượng mẫu")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(bar.get_height())}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "label_distribution.png",
    dpi=300
)
plt.close()

# =====================================
# 6. PHÂN TÍCH ĐỘ DÀI BÌNH LUẬN
# =====================================

df["character_count"] = df["comments"].astype(str).str.len()

df["word_count"] = (
    df["comments"]
    .astype(str)
    .str.split()
    .str.len()
)

length_statistics = df[
    ["character_count", "word_count"]
].describe()

print("\n8. THỐNG KÊ ĐỘ DÀI BÌNH LUẬN")
print(length_statistics)

length_statistics.to_csv(
    TABLE_DIR / "text_length_statistics.csv",
    encoding="utf-8-sig"
)

# Biểu đồ phân bố số từ
plt.figure(figsize=(9, 5))

plt.hist(
    df["word_count"],
    bins=25,
    edgecolor="black"
)

plt.title("Phân bố độ dài bình luận UIT-VSFC")
plt.xlabel("Số từ trong bình luận")
plt.ylabel("Số lượng bình luận")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "word_count_distribution.png",
    dpi=300
)
plt.close()

# =====================================
# 7. ĐỘ DÀI THEO TỪNG NHÃN
# =====================================

length_by_label = (
    df.groupby("sentiment_label")["word_count"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print("\n9. ĐỘ DÀI BÌNH LUẬN THEO NHÃN")
print(length_by_label)

length_by_label.to_csv(
    TABLE_DIR / "length_by_label.csv",
    encoding="utf-8-sig"
)

# =====================================
# 8. LƯU BÁO CÁO TỔNG HỢP
# =====================================

summary = {
    "total_samples": len(df),
    "total_columns": len(df.columns),
    "missing_cells": int(df.isnull().sum().sum()),
    "duplicate_comments": int(duplicate_text),
    "duplicate_rows": int(duplicate_rows),
    "conflicting_label_comments": conflict_count,
    "average_word_count": round(df["word_count"].mean(), 2),
    "median_word_count": round(df["word_count"].median(), 2)
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
)

summary_df.to_csv(
    TABLE_DIR / "eda_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("HOÀN THÀNH KHÁM PHÁ DỮ LIỆU")
print("=" * 60)

print(summary_df.to_string(index=False))

print("\nCác bảng thống kê được lưu tại:", TABLE_DIR)
print("Các biểu đồ được lưu tại:", FIGURE_DIR)
