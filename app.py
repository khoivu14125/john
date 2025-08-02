import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dữ liệu
df = pd.read_csv("amazon_clean.csv")

st.title("Phân tích đánh giá sản phẩm Amazon")

# Biểu đồ 1: Phân phối đánh giá
st.subheader("Phân phối số lượng đánh giá theo sao")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="star_rating", order=sorted(df["star_rating"].unique()), ax=ax1)
st.pyplot(fig1)

# Biểu đồ 2: Tỷ lệ phần trăm theo Vine
st.subheader("Tỷ lệ phần trăm đánh giá theo chương trình Vine")
fig2, ax2 = plt.subplots()
df["vine"].value_counts(normalize=True).plot.pie(autopct="%1.1f%%", ax=ax2)
ax2.set_ylabel("")
st.pyplot(fig2)

# Biểu đồ 3: Verified Purchase
st.subheader("Số lượng đánh giá theo trạng thái xác thực")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x="verified_purchase", ax=ax3)
st.pyplot(fig3)

# Biểu đồ 4: Đánh giá theo thời gian
st.subheader("Số lượng đánh giá theo thời gian")
df["review_date"] = pd.to_datetime(df["review_date"], errors='coerce')
reviews_per_month = df.resample("M", on="review_date").count()
fig4, ax4 = plt.subplots()
reviews_per_month["review_id"].plot(ax=ax4)
ax4.set_ylabel("Số lượng đánh giá")
st.pyplot(fig4)

# Biểu đồ 5: Top 10 sản phẩm có nhiều đánh giá
st.subheader("Top 10 sản phẩm có nhiều đánh giá nhất")
top_products = df["product_title"].value_counts().nlargest(10)
fig5, ax5 = plt.subplots()
top_products.plot(kind="barh", ax=ax5)
ax5.invert_yaxis()
st.pyplot(fig5)

# WordCloud từ nội dung đánh giá
st.subheader("Từ khóa nổi bật trong đánh giá")
text = " ".join(df["review_body"].dropna().astype(str).values)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig6, ax6 = plt.subplots(figsize=(10, 5))
ax6.imshow(wordcloud, interpolation="bilinear")
ax6.axis("off")
st.pyplot(fig6)
