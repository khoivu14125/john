import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình trang
st.set_page_config(page_title="Phân tích sản phẩm Amazon", layout="wide")
st.title("📊 Phân tích dữ liệu đánh giá sản phẩm Amazon")

# Tải dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_clean.csv")
    return df

df = load_data()

# Kiểm tra tên cột
st.subheader("🧾 Các cột trong dữ liệu:")
st.write(df.columns.tolist())

# Chọn cột điểm đánh giá (tùy thuộc vào tên thật trong file)
rating_col = st.selectbox("🔽 Chọn cột chứa điểm đánh giá:", df.columns)

# Hiển thị dữ liệu
st.subheader("🔍 Dữ liệu gốc:")
st.dataframe(df.head())

# Thống kê mô tả
st.subheader("📌 Thống kê mô tả:")
st.write(df.describe())

# Vẽ phân phối điểm đánh giá
if pd.api.types.is_numeric_dtype(df[rating_col]):
    st.subheader(f"⭐ Phân phối điểm đánh giá: {rating_col}")
    fig, ax = plt.subplots()
    sns.histplot(df[rating_col], kde=True, bins=10, ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠️ Cột bạn chọn không phải kiểu số. Hãy chọn một cột số để vẽ biểu đồ.")
