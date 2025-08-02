import streamlit as st
import pandas as pd

# Load dữ liệu đã làm sạch
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_clean.csv")
    return df

df = load_data()

st.title("🔍 Dự đoán điểm đánh giá sản phẩm Amazon")

# Hiển thị data
if df.empty:
    st.error("❌ Không đủ dữ liệu hợp lệ để hiển thị.")
else:
    st.success(f"Dữ liệu chứa {df.shape[0]} dòng và {df.shape[1]} cột.")
    st.dataframe(df.head())
