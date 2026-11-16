# Tf - IDF IDF+ Unigram Unigram+ Bigram 
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "alternative" / "processed"
MODEL_DIR = ROOT / "models" / "uit_vsfc"
RESULT_DIR = ROOT / "results" / "tables" / "uit_vsfc"

TRAIN_FILE = DATA_DIR / "uit_vsfc_train.csv"
TEST_FILE = DATA_DIR / "uit_vsfc_test.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

LABEL_NAMES = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

# Hai cấu hình TF-IDF cần thử nghiệm
TFIDF_CONFIGS = {
    "unigram": {
        "ngram_range": (1, 1)
    },
    "unigram_bigram": {
        "ngram_range": (1, 2)
    }
}

COMMON_PARAMS = {
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 5000,
    "sublinear_tf": True,
    "lowercase": False,
    "strip_accents": None,
    "dtype": np.float64
}


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    print("=" * 60)
    print("ĐỌC DỮ LIỆU UIT-VSFC")
    print("=" * 60)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    required_columns = {
        "segmented_comments",
        "sentiments"
    }

    if not required_columns.issubset(train_df.columns):
        raise ValueError("Train thiếu cột cần thiết.")

    if not required_columns.issubset(test_df.columns):
        raise ValueError("Test thiếu cột cần thiết.")

    X_train = train_df["segmented_comments"].fillna("").astype(str)
    X_test = test_df["segmented_comments"].fillna("").astype(str)

    y_train = train_df["sentiments"].astype(int)
    y_test = test_df["sentiments"].astype(int)

    if set(y_train.unique()) != set(LABEL_NAMES):
        raise ValueError("Nhãn Train không hợp lệ.")

    if set(y_test.unique()) != set(LABEL_NAMES):
        raise ValueError("Nhãn Test không hợp lệ.")

    print(f"Số mẫu Train: {len(X_train)}")
    print(f"Số mẫu Test : {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ==========================================
# 3. BUILD TF-IDF
# ==========================================

def build_tfidf(X_train, X_test, config_name, config):

    print("\n" + "=" * 60)
    print(f"TF-IDF: {config_name.upper()}")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        **COMMON_PARAMS,
        **config
    )

    # Chỉ fit trên Train để tránh data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Test chỉ được transform
    X_test_tfidf = vectorizer.transform(X_test)

    print("Train shape:", X_train_tfidf.shape)
    print("Test shape :", X_test_tfidf.shape)
    print("Số đặc trưng:", len(vectorizer.get_feature_names_out()))

    print(
        "Số phần tử khác 0 trung bình mỗi mẫu Train:",
        round(X_train_tfidf.getnnz(axis=1).mean(), 2)
    )

    return vectorizer, X_train_tfidf, X_test_tfidf


# ==========================================
# 4. SAVE RESULTS
# ==========================================

def save_results(
    config_name,
    vectorizer,
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test
):

    prefix = f"uit_vsfc_{config_name}"

    # Lưu ma trận TF-IDF
    save_npz(
        DATA_DIR / f"{prefix}_X_train.npz",
        X_train_tfidf
    )

    save_npz(
        DATA_DIR / f"{prefix}_X_test.npz",
        X_test_tfidf
    )

    # Lưu vectorizer
    joblib.dump(
        vectorizer,
        MODEL_DIR / f"{prefix}_vectorizer.pkl"
    )

    # Lưu nhãn tương ứng
    pd.DataFrame({
        "sentiments": y_train
    }).to_csv(
        DATA_DIR / f"{prefix}_y_train.csv",
        index=False
    )

    pd.DataFrame({
        "sentiments": y_test
    }).to_csv(
        DATA_DIR / f"{prefix}_y_test.csv",
        index=False
    )

    # Lưu danh sách đặc trưng
    feature_names = vectorizer.get_feature_names_out()

    pd.DataFrame({
        "feature": feature_names,
        "feature_index": range(len(feature_names))
    }).to_csv(
        RESULT_DIR / f"{prefix}_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nĐã lưu cấu hình:", config_name)


# ==========================================
# 5. MAIN
# ==========================================

def main():

    X_train, X_test, y_train, y_test = load_data()

    summary = []

    for config_name, config in TFIDF_CONFIGS.items():

        vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf(
            X_train,
            X_test,
            config_name,
            config
        )

        save_results(
            config_name,
            vectorizer,
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test
        )

        summary.append({
            "configuration": config_name,
            "ngram_range": str(config["ngram_range"]),
            "train_samples": X_train_tfidf.shape[0],
            "test_samples": X_test_tfidf.shape[0],
            "feature_count": X_train_tfidf.shape[1],
            "train_nonzero_avg": round(
                X_train_tfidf.getnnz(axis=1).mean(), 2
            ),
            "test_nonzero_avg": round(
                X_test_tfidf.getnnz(axis=1).mean(), 2
            )
        })

    summary_df = pd.DataFrame(summary)

    summary_path = RESULT_DIR / "tfidf_comparison.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("HOÀN THÀNH TRÍCH XUẤT TF-IDF")
    print("=" * 60)

    print(summary_df.to_string(index=False))
    print("\nBảng so sánh:", summary_path)


if __name__ == "__main__":
    main()
