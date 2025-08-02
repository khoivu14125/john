# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Phân tích sản phẩm Amazon", layout="wide")
st.title("📊 Phân tích dữ liệu đánh giá sản phẩm Amazon")

# Tải dữ liệu
df = pd.read_csv("amazon_clean.csv")

# Hiển thị bảng dữ liệu
st.subheader("🔍 Dữ liệu gốc")
st.dataframe(df)

# Thống kê mô tả
st.subheader("📌 Thống kê mô tả")
st.write(df.describe())

# Phân phối của điểm đánh giá
st.subheader("⭐ Phân phối điểm đánh giá")
fig, ax = plt.subplots()
sns.histplot(df['Rating'], kde=True, bins=10, ax=ax)
st.pyplot(fig)

# Biểu đồ hộp các cột số
st.subheader("📦 Biểu đồ hộp (Boxplot) các đặc trưng số")
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df[num_cols], ax=ax2)
st.pyplot(fig2)

# Biểu đồ tương quan
st.subheader("🔗 Ma trận tương quan")
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# Biểu đồ tương tác (Plotly)
st.subheader("🧭 Biểu đồ tương tác (Plotly)")
if "Rating" in df.columns:
    fig4 = px.scatter(df, x='Rating', y=num_cols[1], color='Rating', title='Tương quan các đặc trưng')
    st.plotly_chart(fig4)

# Tùy chọn lọc dữ liệu
st.subheader("🔍 Lọc dữ liệu theo điểm đánh giá")
rating_filter = st.slider("Chọn khoảng điểm đánh giá", float(df["Rating"].min()), float(df["Rating"].max()), (1.0, 5.0))
filtered_df = df[(df["Rating"] >= rating_filter[0]) & (df["Rating"] <= rating_filter[1])]
st.dataframe(filtered_df)
