
from pathlib import Path
import json
import warnings
import gc

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "alternative" / "processed"
    / "uit_vsfc_train.csv"
)

RESULT_DIR = (
    ROOT / "results" / "tables" / "uit_vsfc"
    / "advanced" / "chi2"
)

MODEL_DIR = (
    ROOT / "models" / "uit_vsfc"
    / "advanced" / "chi2"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 75)
print("THỰC NGHIỆM CHI-SQUARE FEATURE SELECTION - UIT-VSFC")
print("=" * 75)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu: {TRAIN_PATH}"
    )

df = pd.read_csv(TRAIN_PATH)

if df[["segmented_comments", "sentiments"]].isnull().any().any():
    raise ValueError("Dữ liệu có giá trị thiếu.")

X = df["segmented_comments"].astype(str)
y = df["sentiments"].astype(int)

if set(y.unique()) != {0, 1, 2}:
    raise ValueError("Dữ liệu không có đủ ba nhãn.")

print(f"\nSố mẫu Train: {len(df)}")
print("\nPhân bố nhãn:")
print(y.value_counts().sort_index())

# ============================================================
# 3. CÁC PHƯƠNG PHÁP BIỂU DIỄN
# ============================================================

def word_vectorizer(ngram_range):
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        max_features=10000,
        sublinear_tf=True,
        lowercase=False,
        strip_accents=None
    )


def character_vectorizer():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.95,
        max_features=15000,
        sublinear_tf=True,
        lowercase=False,
        strip_accents=None
    )


FEATURE_CONFIGS = {
    "CHI1_Word_Unigram_Bigram": word_vectorizer((1, 2)),

    "CHI2_Word_Unigram_Bigram_Trigram": word_vectorizer((1, 3)),

    "CHI3_Word_Character_LR": FeatureUnion([
        ("word", word_vectorizer((1, 2))),
        ("character", character_vectorizer())
    ]),

    "CHI4_Word_Character_SVM": FeatureUnion([
        ("word", word_vectorizer((1, 2))),
        ("character", character_vectorizer())
    ])
}

# ============================================================
# 4. THUẬT TOÁN VÀ SIÊU THAM SỐ
# ============================================================

CLASSIFIERS = {
    "LogisticRegression": {
        "estimator": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__C": [0.1, 1, 5],
            "classifier__class_weight": [None, "balanced"]
        }
    },

    "LinearSVM": {
        "estimator": LinearSVC(
            random_state=RANDOM_STATE,
            max_iter=20000
        ),
        "params": {
            "classifier__C": [0.1, 0.5, 1, 2],
            "classifier__class_weight": [None, "balanced"]
        }
    }
}

# Chỉ thử các tổ hợp đã được đề xuất
EXPERIMENTS = [
    ("CHI1_Word_Unigram_Bigram", "LogisticRegression"),
    ("CHI1_Word_Unigram_Bigram", "LinearSVM"),
    ("CHI2_Word_Unigram_Bigram_Trigram", "LogisticRegression"),
    ("CHI3_Word_Character_LR", "LogisticRegression"),
    ("CHI4_Word_Character_SVM", "LinearSVM")
]

# k="all": đối chứng không loại bỏ đặc trưng
K_VALUES = [300, 500, 800, 1000, "all"]

# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scorer = make_scorer(
    f1_score,
    average="macro",
    zero_division=0
)

all_results = []

for index, (feature_name, model_name) in enumerate(
    EXPERIMENTS, start=1
):

    print("\n" + "=" * 75)
    print(f"THỰC NGHIỆM CHI2 {index}/{len(EXPERIMENTS)}")
    print(f"Feature: {feature_name}")
    print(f"Model: {model_name}")
    print("=" * 75)

    feature_transformer = FEATURE_CONFIGS[feature_name]
    model_config = CLASSIFIERS[model_name]

    pipeline = Pipeline([
        ("features", feature_transformer),

        # Chi-square được fit riêng trong từng Fold
        ("selection", SelectKBest(score_func=chi2)),

        ("classifier", model_config["estimator"])
    ])

    param_grid = {
        "selection__k": K_VALUES,
        **model_config["params"]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=True,
        error_score="raise"
    )

    search.fit(X, y)

    best_model = search.best_estimator_

    best_k = search.best_params_["selection__k"]

    feature_count = best_model.named_steps[
        "selection"
    ].get_support().sum()

    total_features = best_model.named_steps[
        "features"
    ].transform(X.iloc[:1]).shape[1]

    safe_name = f"{feature_name}_{model_name}"

    model_path = MODEL_DIR / f"{safe_name}.joblib"

    joblib.dump(best_model, model_path)

    # Lưu toàn bộ kết quả tìm kiếm
    detail_df = pd.DataFrame(search.cv_results_)

    detail_df.to_csv(
        RESULT_DIR / f"grid_{safe_name}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result = {
        "feature": feature_name,
        "model": model_name,
        "cv_macro_f1": search.best_score_,
        "selected_k": str(best_k),
        "selected_features": int(feature_count),
        "total_features": int(total_features),
        "best_params": json.dumps(
            search.best_params_,
            ensure_ascii=False,
            default=str
        ),
        "model_path": str(model_path)
    }

    all_results.append(result)

    print(f"Best CV Macro F1: {search.best_score_:.4f}")
    print(f"Best k: {best_k}")
    print(f"Selected features: {feature_count}/{total_features}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Đã lưu mô hình: {model_path}")

    del search
    del pipeline
    del best_model
    gc.collect()

# ============================================================
# 6. TỔNG HỢP KẾT QUẢ
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    "cv_macro_f1",
    ascending=False
).reset_index(drop=True)

results_df.insert(
    0,
    "rank",
    range(1, len(results_df) + 1)
)

results_df["cv_macro_f1"] = (
    results_df["cv_macro_f1"].round(4)
)

comparison_path = (
    RESULT_DIR / "chi2_cv_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 7. XUẤT BẢNG SO SÁNH CÁC MỨC K
# ============================================================

detail_files = list(RESULT_DIR.glob("grid_*.csv"))

detail_tables = []

for file_path in detail_files:
    detail = pd.read_csv(file_path)

    columns = [
        col for col in [
            "param_selection__k",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "rank_test_score",
            "param_classifier__C",
            "param_classifier__class_weight"
        ]
        if col in detail.columns
    ]

    detail = detail[columns].copy()
    detail.insert(0, "experiment", file_path.stem)

    detail_tables.append(detail)

if detail_tables:
    k_comparison = pd.concat(
        detail_tables,
        ignore_index=True
    )

    k_comparison.to_csv(
        RESULT_DIR / "chi2_k_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ============================================================
# 8. HIỂN THỊ KẾT QUẢ
# ============================================================

print("\n" + "=" * 100)
print("BẢNG XẾP HẠNG THỰC NGHIỆM CHI-SQUARE")
print("=" * 100)

print(
    results_df[
        [
            "rank",
            "feature",
            "model",
            "cv_macro_f1",
            "selected_k",
            "selected_features",
            "total_features"
        ]
    ].to_string(index=False)
)

best = results_df.iloc[0]

print("\n" + "=" * 75)
print("MÔ HÌNH CHI-SQUARE TỐT NHẤT")
print("=" * 75)

print(f"Feature: {best['feature']}")
print(f"Model: {best['model']}")
print(f"CV Macro F1: {best['cv_macro_f1']}")
print(f"Selected k: {best['selected_k']}")
print(f"Model path: {best['model_path']}")

print("\n" + "=" * 75)
print("HOÀN THÀNH THỰC NGHIỆM CHI-SQUARE")
print("=" * 75)

print(f"Bảng kết quả: {comparison_path}")
print(f"Thư mục mô hình: {MODEL_DIR}")
print(f"Thư mục kết quả: {RESULT_DIR}")

print(
    "\nLưu ý: Chỉ sử dụng Train 800 mẫu và "
    "Stratified 5-Fold Cross-validation."
)
print("Không sử dụng Test 200 hoặc Official Test.")
