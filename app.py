import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("📊 Ứng dụng Phân tích và Dự báo Đánh giá Sản phẩm Amazon")

# Bước 1: Tải dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_clean.csv")
    return df

df = load_data()

# Hiển thị thông tin cơ bản
st.subheader("🔍 Bước 1: Cấu trúc & Mô tả Dữ liệu")
st.write("Số dòng:", df.shape[0])
st.write("Số cột:", df.shape[1])
st.write("Tên các cột:", df.columns.tolist())
st.dataframe(df.head())

# Bước 2: Tiền xử lý
st.subheader("🧹 Bước 2: Tiền xử lý dữ liệu")
df = df.dropna()
df = df.drop_duplicates()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
st.success("✅ Đã xử lý null, trùng lặp và chuẩn hóa dữ liệu số.")
st.dataframe(df.head())

# Bước 3: Phân tích dữ liệu
st.subheader("📌 Bước 3: Phân tích dữ liệu")
selected_col = st.selectbox("Chọn cột để phân tích phân phối:", numeric_cols)
fig1 = px.histogram(df, x=selected_col, nbins=20, title=f"Phân phối {selected_col}")
st.plotly_chart(fig1, use_container_width=True)

# Bước 4: Trực quan hóa
st.subheader("📈 Bước 4: Trực quan hóa dữ liệu")
fig2 = px.scatter_matrix(df[numeric_cols], title="Ma trận phân tán các biến số")
st.plotly_chart(fig2, use_container_width=True)

# Bước 5: Huấn luyện mô hình
st.subheader("🤖 Bước 5: Huấn luyện mô hình dự báo")
target = st.selectbox("Chọn biến mục tiêu (y):", numeric_cols)
features = st.multiselect("Chọn biến đầu vào (X):", [col for col in numeric_cols if col != target])

if features and target:
    X = df[features]
    y = df[target]
    
    if len(df) < 5:
        st.error("❌ Không đủ dữ liệu để huấn luyện mô hình. Cần ít nhất 5 dòng dữ liệu.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.write("Hệ số mô hình:", model.coef_)
        st.write("Sai số trung bình bình phương (MSE):", mean_squared_error(y_test, y_pred))
        st.write("Hệ số R2:", r2_score(y_test, y_pred))

# Bước 6: Đánh giá mô hình
st.subheader("📉 Bước 6: Đánh giá mô hình")
if features and target and len(df) >= 5:
    fig3, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Giá trị thực tế")
    ax.set_ylabel("Giá trị dự đoán")
    ax.set_title("Biểu đồ thực tế vs dự đoán")
    st.pyplot(fig3)
