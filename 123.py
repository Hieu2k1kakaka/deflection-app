# deflection_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import sys
import pickle
from PIL import Image

# Cấu hình trang
st.set_page_config(page_title="Dự đoán Độ Võng Cực Đại", page_icon="🔵", layout="centered")

# CSS nền đẹp
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url('background_beam.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);  /* Mờ để dễ đọc chữ */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Hiển thị logo + tiêu đề
logo = Image.open("logo_transparent.png")
st.image(logo, width=150)
st.markdown("<h1 style='text-align: center;'>🔵 Ứng dụng Dự Đoán Độ Võng Cực Đại</h1>", unsafe_allow_html=True)

# Đường dẫn tương thích cho app khi build exe
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
st.subheader("Nhập thông số dầm:")
b = st.number_input("Bề rộng b (m)", min_value=0.01, value=0.15, step=0.01)
h = st.number_input("Chiều cao h (m)", min_value=0.01, value=0.30, step=0.01)
E = st.number_input("Mô đun đàn hồi E (Pa)", min_value=1e5, value=2.0e11, step=1e9, format="%.1e")
L = st.number_input("Chiều dài dầm L (m)", min_value=0.1, value=3.0, step=0.1)
F = st.number_input("Tải trọng F (N)", min_value=0.0, value=5000.0, step=100.0)

if st.button("Dự đoán độ võng cực đại"):
    input_data = np.array([[b, h, E, L, F]])
    input_scaled = x_scaler.transform(input_data)
    delta_scaled = model.predict(input_scaled)
    delta_max_pred = y_scaler.inverse_transform(delta_scaled)[0][0]

    st.success(f"✅ Độ võng cực đại dự đoán là: **{delta_max_pred:.6e} m**")

    # Vẽ hình ảnh mô phỏng
    x = np.linspace(0, L, 100)
    y = -(F * x**2) / (6 * E * (b * h**3)) * (3 * L - x)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, y, color='blue', linewidth=3)
    ax.set_xlabel("Chiều dài dầm (m)")
    ax.set_ylabel("Độ võng (m)")
    ax.set_title("Mô hình Dầm Cantilever Bị Võng")
    ax.grid(True)
    ax.set_xlim(0, L)
    ax.set_ylim(1.5 * np.min(y), 0.5 * np.max(y))

    st.pyplot(fig)
