
# Student Sentiment Analysis – UIT-VSFC

## 1. Giới thiệu

**Tên đề tài:** Nghiên cứu và xây dựng mô hình phân loại cảm xúc từ phản hồi của sinh viên.

Student Sentiment Analysis là dự án nghiên cứu và ứng dụng Machine Learning trong lĩnh vực xử lý ngôn ngữ tự nhiên (Natural Language Processing – NLP), tập trung vào bài toán phân loại cảm xúc từ các bình luận, nhận xét của sinh viên về hoạt động giảng dạy và học tập.

Dự án sử dụng bộ dữ liệu UIT-VSFC (Vietnamese Students' Feedback Corpus) để nghiên cứu khả năng nhận diện và phân loại cảm xúc trong văn bản tiếng Việt.

Hệ thống phân loại cảm xúc thành ba nhóm:

| Nhãn | Giá trị | Ý nghĩa |
|---|---:|---|
| Negative | 0 | Tiêu cực |
| Positive | 1 | Tích cực |
| Neutral | 2 | Trung tính |

## 2. Mục tiêu nghiên cứu

- Khám phá và đánh giá chất lượng dữ liệu phản hồi sinh viên.
- Xây dựng quy trình tiền xử lý văn bản tiếng Việt.
- Nghiên cứu phương pháp biểu diễn văn bản bằng TF-IDF.
- Thực nghiệm và so sánh các thuật toán Machine Learning.
- Tối ưu hóa tham số mô hình bằng GridSearchCV.
- Đánh giá hiệu quả phân loại bằng Accuracy, Precision, Recall và F1-score.
- Phân tích lỗi và xác định những hạn chế của mô hình.
- Xây dựng mô hình dự đoán cảm xúc từ các bình luận tiếng Việt mới.

## 3. Dataset

### 3.1. Nguồn dữ liệu

Nghiên cứu sử dụng bộ dữ liệu UIT-VSFC:

- Dataset: [UIT-VSFC trên Hugging Face](https://huggingface.co/datasets/ura-hcmut/UIT-VSFC)
- Ngôn ngữ: Tiếng Việt.
- Lĩnh vực: Phản hồi của sinh viên.
- Bài toán: Phân loại cảm xúc văn bản.

### 3.2. Phạm vi dữ liệu nghiên cứu

Nghiên cứu lựa chọn 1.000 mẫu từ tập Train chính thức của UIT-VSFC để xây dựng bộ dữ liệu thực nghiệm cân bằng.

Phân bố nhãn:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 333 | 33,3% |
| Positive | 333 | 33,3% |
| Neutral | 334 | 33,4% |
| **Tổng** | **1.000** | **100%** |

Dữ liệu được chia thành:

| Tập dữ liệu | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Training | 800 | 80% |
| Testing | 200 | 20% |
| **Tổng** | **1.000** | **100%** |

Phương pháp chia dữ liệu:

- Stratified Train-Test Split.
- Test size: 0.2.
- Random State: 42.
- Bảo toàn tỷ lệ phân bố nhãn giữa hai tập.

Phân bố sau khi chia:

| Nhãn | Train (800) | Test (200) |
|---|---:|---:|
| Negative | 266 | 67 |
| Positive | 267 | 66 |
| Neutral | 267 | 67 |

### 3.3. Official Test

Ngoài bộ dữ liệu thực nghiệm 1.000 mẫu, nghiên cứu sử dụng tập Test chính thức gồm 3.166 mẫu của UIT-VSFC để đánh giá bổ sung.

Phân bố nhãn:

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Negative | 1.409 | 44,50% |
| Positive | 1.590 | 50,22% |
| Neutral | 167 | 5,27% |
| **Tổng** | **3.166** | **100%** |

**Lưu ý:** Official Test đã được sử dụng để đánh giá và phân tích lỗi trong quá trình nghiên cứu. Vì vậy, kết quả trên tập này không được xem là đánh giá độc lập hoàn toàn chưa từng được quan sát.

## 4. Khám phá và tiền xử lý dữ liệu

### 4.1. Khám phá dữ liệu (EDA)

Giai đoạn EDA được thực hiện nhằm khảo sát đặc điểm và chất lượng bộ dữ liệu trước khi xây dựng mô hình.

Các nội dung phân tích:

- Kiểm tra cấu trúc và kiểu dữ liệu.
- Thống kê số lượng mẫu theo từng nhãn.
- Kiểm tra dữ liệu thiếu.
- Phát hiện các bản ghi trùng lặp.
- Phân tích độ dài bình luận.
- Khảo sát phân bố độ dài văn bản theo nhãn.
- Phân tích các từ và cụm từ xuất hiện phổ biến.

### 4.2. Tiền xử lý văn bản

Quy trình tiền xử lý được thiết kế phù hợp với đặc điểm tiếng Việt.

Các bước thực hiện:

1. Kiểm tra và loại bỏ bản ghi thiếu nội dung hoặc nhãn không hợp lệ.
2. Chuẩn hóa Unicode theo NFC.
3. Chuyển văn bản về chữ thường.
4. Giải mã HTML entities.
5. Loại bỏ URL và địa chỉ email.
6. Chuẩn hóa khoảng trắng.
7. Phát hiện và xử lý bản ghi trùng lặp.
8. Phân đoạn từ tiếng Việt bằng Underthesea.

### 4.3. Nguyên tắc tiền xử lý

Nghiên cứu áp dụng các nguyên tắc:

- Giữ nguyên nội dung bình luận gốc trong cột `comments`.
- Tạo cột `clean_comments` để lưu văn bản sau làm sạch.
- Tạo cột `segmented_comments` để lưu văn bản sau phân đoạn từ.
- Giữ nguyên dấu tiếng Việt.
- Không loại bỏ các từ phủ định như "không", "chưa", "chẳng".
- Không loại bỏ từ dừng hàng loạt.
- Không thực hiện stemming hoặc lemmatization.
- Bảo toàn các từ và cụm từ có ý nghĩa trong ngữ cảnh phản hồi sinh viên.

Việc giữ lại từ phủ định giúp hạn chế thay đổi ý nghĩa cảm xúc của câu sau tiền xử lý.

## 5. Phương pháp nghiên cứu

### 5.1. Quy trình tổng thể

```text
UIT-VSFC Dataset (1,000 samples)
              |
              v
       Data Exploration
              |
              v
        Data Cleaning
              |
              v
 Vietnamese Word Segmentation
              |
              v
        Train-Test Split
              |
              v
       TF-IDF Extraction
              |
              v
        Model Training
              |
              v
    Hyperparameter Optimization
              |
              v
       Model Evaluation
              |
              v
       Final Model Selection
              |
              v
      Sentiment Prediction
```

### 5.2. Trích xuất đặc trưng TF-IDF

Nghiên cứu sử dụng TF-IDF (Term Frequency – Inverse Document Frequency) để chuyển đổi văn bản thành biểu diễn số học phục vụ huấn luyện mô hình.

Các phương pháp đặc trưng đã thử nghiệm:

| Phương pháp | Mô tả |
|---|---|
| Word Unigram | Biểu diễn từng từ riêng lẻ |
| Word Unigram + Bigram | Kết hợp từ đơn và cụm hai từ |
| Word Unigram + Bigram + Trigram | Kết hợp từ đơn, cụm hai từ và ba từ |
| Character N-gram | Biểu diễn các chuỗi ký tự |
| Word + Character | Kết hợp đặc trưng từ và ký tự |

TF-IDF được tích hợp bên trong Pipeline để quá trình học từ vựng và trọng số IDF được thực hiện riêng trong từng fold của Cross-validation, hạn chế rò rỉ dữ liệu.

### 5.3. Các thuật toán thực nghiệm

Nghiên cứu đã thử nghiệm:

1. Multinomial Naive Bayes.
2. Complement Naive Bayes.
3. Logistic Regression.
4. Linear Support Vector Machine (Linear SVM).
5. Random Forest.

Các thử nghiệm bổ sung:

- So sánh Word N-gram và Character N-gram.
- Kết hợp Word và Character Features.
- Tối ưu siêu tham số bằng GridSearchCV.
- Lựa chọn đặc trưng bằng Chi-square.
- So sánh hiệu quả giữa các cấu hình mô hình.

### 5.4. Phương pháp tối ưu

Sử dụng Stratified K-Fold Cross-validation với 5 folds.

Macro F1-score được chọn làm chỉ số chính trong quá trình lựa chọn mô hình vì chỉ số này đánh giá hiệu quả trên từng lớp với trọng số ngang nhau.

## 6. Lựa chọn mô hình cuối cùng

Sau quá trình thực nghiệm, mô hình được lựa chọn là:

**Logistic Regression + TF-IDF Word Unigram + Bigram.**

Cấu hình chính:

| Thành phần | Cấu hình |
|---|---|
| Feature Extraction | TfidfVectorizer |
| Analyzer | Word |
| N-gram Range | (1, 2) |
| Min DF | 2 |
| Max DF | 0.95 |
| Max Features | 10.000 |
| Sublinear TF | True |
| Classifier | Logistic Regression |
| C | 1 |
| Class Weight | Balanced |
| Random State | 42 |
| Max Iterations | 5.000 |

Hai phương án có và không sử dụng Chi-square được so sánh trên cùng các fold. Kết quả không ghi nhận sự cải thiện điểm số khi bổ sung Chi-square trong cấu hình đã thử nghiệm.

Do đó, phương án không sử dụng Chi-square được lựa chọn vì có cấu trúc đơn giản hơn.

Mô hình cuối được lưu tại:

```text
models/uit_vsfc/final/
├── uit_vsfc_final_model.joblib
└── final_model_metadata.json
```

## 7. Kết quả thực nghiệm

### 7.1. Kết quả Cross-validation

Mô hình cuối đạt:

| Chỉ số | Kết quả |
|---|---:|
| Accuracy trung bình | 77,87% |
| Macro F1 trung bình | 77,85% |
| Độ lệch chuẩn Macro F1 | 2,81% |

Điểm Macro F1 qua 5 folds:

| Fold | Macro F1 |
|---|---:|
| Fold 1 | 73,76% |
| Fold 2 | 80,08% |
| Fold 3 | 78,31% |
| Fold 4 | 76,46% |
| Fold 5 | 80,62% |

### 7.2. Kết quả trên Official Test

Mô hình được đánh giá bổ sung trên 3.166 mẫu Official Test.

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | 75,43% |
| Macro Precision | 65,18% |
| Macro Recall | 71,24% |
| Macro F1-score | 63,55% |
| Weighted F1-score | 79,40% |

Kết quả theo từng lớp:

| Nhãn | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 85,99% | 73,17% | 79,06% |
| Positive | 92,82% | 78,87% | 85,28% |
| Neutral | 16,72% | 61,68% | 26,31% |

### 7.3. Phân tích kết quả

Mô hình nhận diện tương đối tốt hai lớp Negative và Positive. Tuy nhiên, khả năng phân loại lớp Neutral còn hạn chế.

Các vấn đề được ghi nhận:

- Lớp Neutral có số lượng mẫu thấp trong Official Test.
- Mô hình dự đoán Neutral nhiều hơn số lượng thực tế của lớp này.
- Một phần đáng kể lỗi phân loại xuất phát từ việc các mẫu Negative và Positive bị dự đoán thành Neutral.
- Sự khác biệt về phân bố nhãn giữa tập huấn luyện cân bằng và Official Test không cân bằng có thể ảnh hưởng đến khả năng tổng quát hóa.

Đây là những hạn chế cần được thảo luận trong báo cáo nghiên cứu và xem xét trong các hướng cải tiến tiếp theo.

## 8. Cấu trúc dự án

```text
StudentSentimentAnalysis/
│
├── data/
│   └── alternative/
│       ├── uit_vsfc_1000.csv
│       ├── uit_vsfc_1000.xlsx
│       │
│       ├── processed/
│       │   ├── uit_vsfc_preprocessed.csv
│       │   ├── uit_vsfc_segmented.csv
│       │   ├── uit_vsfc_train.csv
│       │   └── uit_vsfc_test.csv
│       │
│       └── official_test/
│           └── uit_vsfc_official_test_processed.csv
│
├── src/
│   ├── analysis/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── prediction/
│
├── models/
│   └── uit_vsfc/
│       ├── advanced/
│       ├── pipeline/
│       └── final/
│           ├── uit_vsfc_final_model.joblib
│           └── final_model_metadata.json
│
├── results/
│   ├── figures/
│   │   └── uit_vsfc/
│   └── tables/
│       └── uit_vsfc/
│
├── notebooks/
├── archive/
├── main.py
├── requirements.txt
└── README.md
```

## 9. Hướng dẫn cài đặt và sử dụng

### 9.1. Yêu cầu môi trường

- Python.
- Pandas.
- NumPy.
- Scikit-learn.
- SciPy.
- Underthesea.
- Matplotlib.
- Joblib.
- Openpyxl.

### 9.2. Cài đặt thư viện

```bash
python -m pip install -r requirements.txt
```

### 9.3. Chạy chương trình dự đoán

Tại thư mục gốc dự án:

```bash
python src/prediction/predict_sentiment.py
```

Chương trình cho phép nhập bình luận tiếng Việt, thực hiện tiền xử lý, phân đoạn từ và trả về:

- Nhãn cảm xúc dự đoán.
- Xác suất dự đoán của từng lớp.
- Văn bản sau tiền xử lý.

Nhập `exit` để kết thúc chương trình.

### 9.4. Chạy các bước nghiên cứu

Các script được tổ chức theo chức năng trong thư mục `src/`.

Ví dụ:

```bash
# Khám phá dữ liệu
python src/analysis/explore_uit_vsfc.py

# Tiền xử lý dữ liệu
python src/analysis/preprocess_uit_vsfc.py

# Phân đoạn từ
python src/data/segment_uit_vsfc.py

# Chia tập dữ liệu
python src/data/split_uit_vsfc.py
```

Các chương trình huấn luyện và đánh giá được chạy riêng theo từng giai đoạn.

**Lưu ý:** Một số script có thể tạo hoặc ghi đè file trong `data/`, `models/` và `results/`. Nên sao lưu dữ liệu trước khi chạy lại các thí nghiệm.

## 10. Hạn chế của nghiên cứu

Nghiên cứu hiện tại có một số hạn chế:

- Bộ dữ liệu thực nghiệm 1.000 mẫu có quy mô tương đối nhỏ.
- Dữ liệu huấn luyện được lấy mẫu cân bằng, trong khi phân bố nhãn của Official Test không cân bằng.
- Khả năng nhận diện lớp Neutral chưa cao.
- TF-IDF chủ yếu biểu diễn tần suất và trọng số từ, chưa nắm bắt đầy đủ ngữ nghĩa và ngữ cảnh phức tạp.
- Các câu có cấu trúc phủ định hoặc biểu đạt cảm xúc gián tiếp có thể gây nhầm lẫn.
- Official Test đã được quan sát trong quá trình nghiên cứu, do đó cần thêm tập đánh giá độc lập để kiểm chứng khả năng tổng quát hóa.

## 11. Hướng phát triển

Các hướng nghiên cứu tiếp theo:

1. Mở rộng quy mô dữ liệu và tăng tính đa dạng của phản hồi sinh viên.
2. Kiểm tra và xác minh chất lượng nhãn bằng phương pháp gán nhãn thủ công.
3. Nghiên cứu xử lý phủ định và biểu đạt cảm xúc theo ngữ cảnh.
4. Thử nghiệm Sentence Transformer.
5. Nghiên cứu các mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt.
6. Cải thiện khả năng nhận diện lớp Neutral.
7. Xây dựng giao diện trực quan để hỗ trợ nhập bình luận và phân tích kết quả cảm xúc.
8. Đánh giá mô hình trên tập dữ liệu độc lập chưa được sử dụng trong quá trình lựa chọn mô hình.

## 12. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình |
| Pandas | Xử lý dữ liệu |
| NumPy | Tính toán số học |
| Scikit-learn | Xây dựng và đánh giá mô hình Machine Learning |
| Underthesea | Phân đoạn từ tiếng Việt |
| SciPy | Hỗ trợ xử lý và tính toán |
| Matplotlib | Trực quan hóa dữ liệu |
| Joblib | Lưu và tải mô hình |
| Jupyter Notebook | Khám phá và thực nghiệm |

## 13. Mục đích sử dụng

Dự án được xây dựng phục vụ nghiên cứu khoa học và thực nghiệm trong lĩnh vực Natural Language Processing, tập trung vào bài toán phân tích cảm xúc từ phản hồi của sinh viên bằng các phương pháp Machine Learning.

Kết quả nghiên cứu là cơ sở để tìm hiểu khả năng ứng dụng các kỹ thuật xử lý ngôn ngữ tự nhiên trong việc khai thác, phân tích và hỗ trợ đánh giá phản hồi của người học.

---

**Student Sentiment Analysis – Vietnamese Student Feedback Classification**

*Research project using UIT-VSFC dataset and Machine Learning methods.*
