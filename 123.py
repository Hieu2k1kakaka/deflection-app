# fD_predict_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import sys
import pickle
from PIL import Image
import base64

# Cấu hình trang
st.set_page_config(page_title="Dự đoán fD", page_icon="🔵", layout="centered")

# Hàm đọc ảnh nền
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Đặt ảnh nền và khung
image_base64 = get_base64_image("logo_transparent.jpg")
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .stApp {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 20px;
        max-width: 900px;
        margin: auto;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
    }}
    h1, h2, h3, .stButton>button {{
        color: #002B5B;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Tiêu đề
st.markdown("<h1 style='text-align: center;'>🔵 Ứng dụng Dự Đoán fD</h1>", unsafe_allow_html=True)

# Hàm hỗ trợ đường dẫn tương thích
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load model và scaler
model = keras.models.load_model(resource_path('fnn_deflection_model.h5'))
with open(resource_path('x_scaler.pkl'), 'rb') as f:
    x_scaler = pickle.load(f)
with open(resource_path('y_scaler.pkl'), 'rb') as f:
    y_scaler = pickle.load(f)

# Giao diện nhập liệu
st.subheader("Nhập thông số vật liệu và điều kiện:")
muy = st.number_input("μ (Tham số phi địa phương)", min_value=0.0, value=0.5, step=0.01)
v = st.number_input("v (Vận tốc lực - m/s)", min_value=0.0, value=1.0, step=0.1)
n = st.number_input("n (Tham số vật liệu FMG)", min_value=0.0, value=2.0, step=0.1)
DeltaT = st.number_input("ΔT (Chênh lệch nhiệt độ - °C)", min_value=0.0, value=50.0, step=1.0)

# Dự đoán và hiển thị
if st.button("Dự đoán fD"):
    input_data = np.array([[muy, v, n, DeltaT]])
    input_scaled = x_scaler.transform(input_data)
    fD_scaled = model.predict(input_scaled)
    fD_pred = y_scaler.inverse_transform(fD_scaled)[0][0]

    st.success(f"✅ fD dự đoán là: **{fD_pred:.6f}**")

# Phần so sánh với dữ liệu thực tế từ file 'nanotable.csv'
st.markdown("---")
st.subheader("📂 So sánh toàn bộ fD thực tế và fD dự đoán từ file `nanobeam.csv`")

try:
    file_path = os.path.join(os.path.dirname(__file__), 'nanotable.csv')
    df = pd.read_csv(file_path)

    required_cols = {"muy", "v", "n", "DeltaT", "fD"}
    if not required_cols.issubset(df.columns):
        st.error("❌ File CSV cần chứa các cột: muy, v, n, DeltaT, fD")
    else:
        st.success("✅ Đọc dữ liệu thành công từ nanobeam.csv")
        st.dataframe(df.head())

        # Dự đoán fD từ mô hình
        X_data = df[["muy", "v", "n", "DeltaT"]].values
        X_scaled = x_scaler.transform(X_data)
        fD_pred_scaled = model.predict(X_scaled)
        fD_pred = y_scaler.inverse_transform(fD_pred_scaled).flatten()

        # Gộp kết quả
        df["fD_dudoan"] = fD_pred
        df["sai_so"] = np.abs(df["fD"] - df["fD_dudoan"])

        st.markdown("### 🧾 Bảng so sánh fD thực tế vs. fD dự đoán")
        st.dataframe(df)

        # Vẽ biểu đồ fD
        fig3, ax3 = plt.subplots()
        ax3.plot(df["fD"], label="fD thực tế", marker='o')
        ax3.plot(df["fD_dudoan"], label="fD dự đoán", marker='x')
        ax3.set_xlabel("Chỉ số mẫu")
        ax3.set_ylabel("Giá trị fD")
        ax3.set_title("So sánh fD thực tế và dự đoán")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

        # Vẽ sai số
        fig4, ax4 = plt.subplots()
        ax4.plot(df["sai_so"], marker='d', color='red')
        ax4.set_xlabel("Chỉ số mẫu")
        ax4.set_ylabel("Sai số tuyệt đối")
        ax4.set_title("Biểu đồ sai số fD")
        ax4.grid(True)
        st.pyplot(fig4)

except Exception as e:
    st.error(f"❌ Lỗi khi đọc hoặc xử lý file: {e}")


