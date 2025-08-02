import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dữ liệu
df = pd.read_csv("amazon_clean.csv")
df.columns = df.columns.str.strip()

st.title("Phân tích đánh giá sản phẩm Amazon")
st.subheader("🧾 Các cột thực tế trong dữ liệu:")
st.write(df.columns.tolist())

# Biểu đồ 1: Phân phối điểm đánh giá
if "rating" in df.columns:
    st.subheader("Phân phối số lượng đánh giá theo điểm đánh giá (rating)")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="rating", order=sorted(df["rating"].dropna().unique()), ax=ax1)
    st.pyplot(fig1)
else:
    st.warning("Không tìm thấy cột 'rating'")

# Biểu đồ 2: Top 10 sản phẩm được đánh giá nhiều nhất
if "product_name" in df.columns:
    st.subheader("Top 10 sản phẩm có nhiều đánh giá nhất")
    top_products = df["product_name"].value_counts().nlargest(10)
    fig2, ax2 = plt.subplots()
    top_products.plot(kind="barh", ax=ax2)
    ax2.invert_yaxis()
    st.pyplot(fig2)
else:
    st.warning("Không tìm thấy cột 'product_name'")

# Biểu đồ 3: WordCloud từ nội dung đánh giá
if "review_content" in df.columns:
    st.subheader("Từ khóa nổi bật trong đánh giá")
    text = " ".join(df["review_content"].dropna().astype(str).values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation="bilinear")
    ax3.axis("off")
    st.pyplot(fig3)
else:
    st.warning("Không tìm thấy cột 'review_content'")
