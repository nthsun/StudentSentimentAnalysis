
from pathlib import Path
import sys

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prediction.predict_sentiment import load_model, predict_sentiment

st.set_page_config(
    page_title="Phân loại cảm xúc sinh viên",
    page_icon="📊",
    layout="wide"
)

LABELS = {
    0: "Tiêu cực",
    1: "Tích cực",
    2: "Trung tính"
}

COLORS = {
    "Tiêu cực": "#EF4444",
    "Tích cực": "#22C55E",
    "Trung tính": "#64748B"
}


@st.cache_resource
def get_model():
    return load_model()


def main():
    st.title("📊 HỆ THỐNG PHÂN LOẠI CẢM XÚC")
    st.caption("Phân tích phản hồi sinh viên bằng TF-IDF và Logistic Regression | Dataset: UIT-VSFC")

    try:
        model = get_model()
    except Exception as error:
        st.error(f"Không thể tải mô hình: {error}")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    tab1, tab2, tab3 = st.tabs([
        "🔍 Phân tích cảm xúc",
        "📈 Thống kê",
        "ℹ️ Thông tin mô hình"
    ])

    with tab1:
        st.subheader("Nhập phản hồi cần phân tích")

        text = st.text_area(
            "Nội dung phản hồi",
            placeholder="Ví dụ: Giảng viên giảng bài dễ hiểu và nhiệt tình...",
            height=150
        )

        if st.button("Phân tích cảm xúc", type="primary"):
            try:
                if not text.strip():
                    st.warning("Vui lòng nhập nội dung phản hồi.")
                else:
                    result = predict_sentiment(model, text)

                    result["probabilities"] = {
                        LABELS[k]: v
                        for k, v in result["probabilities"].items()
                    }

                    st.session_state.history.insert(0, result)

            except (ValueError, TypeError) as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"Lỗi phân tích: {error}")

        if st.session_state.history:
            result = st.session_state.history[0]

            st.divider()
            st.subheader("Kết quả phân tích")

            sentiment = result["sentiment_vietnamese"]
            st.markdown(f"### Kết quả: {sentiment}")

            cols = st.columns(3)

            for col, label in zip(
                cols,
                ["Tiêu cực", "Tích cực", "Trung tính"]
            ):
                probability = result["probabilities"].get(label, 0)
                col.metric(label, f"{probability * 100:.2f}%")

            st.caption(
                "Các tỷ lệ trên là xác suất dự đoán của mô hình, "
                "không phải mức độ chính xác được bảo đảm."
            )

            st.subheader("Văn bản sau tiền xử lý")
            st.write(result["processed_text"])

            st.subheader("Lịch sử phân tích")
            history_df = pd.DataFrame([
                {
                    "Nội dung": item["original_text"],
                    "Kết quả": item["sentiment_vietnamese"],
                    "Xác suất cao nhất": (
                        f"{max(item['probabilities'].values()) * 100:.2f}%"
                    )
                }
                for item in st.session_state.history
            ])

            st.dataframe(history_df, use_container_width=True)

            csv = history_df.to_csv(
                index=False,
                encoding="utf-8-sig"
            )

            st.download_button(
                "📥 Xuất lịch sử CSV",
                data=csv,
                file_name="sentiment_history.csv",
                mime="text/csv"
            )

            if st.button("Xóa lịch sử"):
                st.session_state.history = []
                st.rerun()

    with tab2:
        st.subheader("Thống kê kết quả phân tích")

        history = st.session_state.history

        if not history:
            st.info("Chưa có dữ liệu. Hãy phân tích phản hồi trước.")
        else:
            df = pd.DataFrame([
                item["sentiment_vietnamese"]
                for item in history
            ], columns=["Cảm xúc"])

            counts = (
                df["Cảm xúc"]
                .value_counts()
                .reindex(
                    ["Tiêu cực", "Tích cực", "Trung tính"],
                    fill_value=0
                )
            )

            c1, c2 = st.columns(2)

            c1.metric("Tổng phản hồi", len(history))
            c2.metric(
                "Cảm xúc phổ biến",
                counts.idxmax() if counts.sum() else "Chưa có"
            )

            st.bar_chart(counts)

            st.dataframe(
                counts.rename("Số lượng"),
                use_container_width=True
            )

    with tab3:
        st.subheader("Thông tin mô hình")

        st.markdown("""
        **Tên hệ thống:** Phân loại cảm xúc phản hồi sinh viên

        **Bộ dữ liệu:** UIT-VSFC

        **Phương pháp trích xuất đặc trưng:** TF-IDF

        **Thuật toán phân loại:** Logistic Regression

        **Ngôn ngữ:** Python

        **Thư viện chính:** Scikit-learn, Underthesea, Joblib, Streamlit

        **Các nhãn phân loại:**
        - 0 – Tiêu cực (Negative)
        - 1 – Tích cực (Positive)
        - 2 – Trung tính (Neutral)
        """)

        st.info(
            "Hệ thống sử dụng mô hình đã huấn luyện và lưu tại "
            "models/uit_vsfc/final/uit_vsfc_final_model.joblib."
        )


if __name__ == "__main__":
    main()
