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


    # Thêm phần vẽ biểu đồ độ võng minh họa
    st.markdown("### 📉 Biểu đồ minh họa độ võng dầm")
    
    L = 1.0  # độ dài dầm giả định (1m)
    x_vals = np.linspace(0, L, 100)
    y_vals = -fD_pred * (x_vals / L)**2 * (3 - 2 * (x_vals / L))  # độ võng theo chiều dài

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x_vals, y_vals, color='blue', linewidth=3)
    ax.set_xlabel("Chiều dài dầm (m)")
    ax.set_ylabel("Độ võng mô phỏng (m)")
    ax.set_title("Đường cong minh họa độ võng của dầm")
    ax.grid(True)
    ax.set_xlim(0, L)
    ax.set_ylim(1.5 * np.min(y_vals), 0.5 * np.max(y_vals))
    st.pyplot(fig)

