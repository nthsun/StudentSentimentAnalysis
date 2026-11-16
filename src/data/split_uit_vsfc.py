
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================
# 1. CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "alternative" /
    "processed" / "uit_vsfc_segmented.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "alternative" / "processed"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

LABEL_NAMES = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}


# =====================================
# 2. LOAD DATA
# =====================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "segmented_comments",
        "sentiments",
        "sentiment_label"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Thiếu cột: {required_columns}"
        )

    df = df.dropna(
        subset=["segmented_comments", "sentiments"]
    ).copy()

    df["sentiments"] = df["sentiments"].astype(int)

    return df


# =====================================
# 3. SPLIT TRAIN / TEST
# =====================================

def split_data(df):

    X = df["segmented_comments"]
    y = df["sentiments"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_df = pd.DataFrame({
        "segmented_comments": X_train,
        "sentiments": y_train
    })

    test_df = pd.DataFrame({
        "segmented_comments": X_test,
        "sentiments": y_test
    })

    train_df["sentiment_label"] = (
        train_df["sentiments"].map(LABEL_NAMES)
    )

    test_df["sentiment_label"] = (
        test_df["sentiments"].map(LABEL_NAMES)
    )

    return train_df, test_df


# =====================================
# 4. SAVE AND REPORT
# =====================================

def save_data(train_df, test_df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "uit_vsfc_train.csv"
    test_path = OUTPUT_DIR / "uit_vsfc_test.csv"

    train_df.to_csv(
        train_path,
        index=False,
        encoding="utf-8-sig"
    )

    test_df.to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nKẾT QUẢ CHIA DỮ LIỆU")
    print("=" * 50)

    print(f"Train: {len(train_df)} mẫu")
    print(f"Test : {len(test_df)} mẫu")

    for name, data in [
        ("TRAIN", train_df),
        ("TEST", test_df)
    ]:
        print(f"\nPhân bố nhãn {name}:")

        counts = (
            data["sentiments"]
            .value_counts()
            .reindex(LABEL_NAMES.keys(), fill_value=0)
        )

        for label_id, count in counts.items():
            print(
                f"{LABEL_NAMES[label_id]:10s}: "
                f"{count:3d} mẫu"
            )

    # Kiểm tra không có bình luận trùng giữa Train và Test
    overlap = set(train_df["segmented_comments"]) & set(
        test_df["segmented_comments"]
    )

    assert len(overlap) == 0, (
        "Phát hiện nội dung trùng giữa Train và Test"
    )

    assert len(train_df) + len(test_df) == len(train_df) + len(test_df)

    print("\nĐã lưu:")
    print(train_path)
    print(test_path)


# =====================================
# 5. MAIN
# =====================================

def main():
    df = load_data()

    print("Tổng số mẫu:", len(df))

    train_df, test_df = split_data(df)

    save_data(train_df, test_df)

    print("\nHOÀN THÀNH CHIA TRAIN/TEST")


if __name__ == "__main__":
    main()
