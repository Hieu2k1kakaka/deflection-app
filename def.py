# deflection_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import sys

# Giao diện
st.set_page_config(page_title="Dự đoán Độ Võng Cực Đại", page_icon="🔵", layout="centered")

# Màu nền xanh da trời
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #d0ebff;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🔵 Dự đoán Độ Võng Cực Đại của Dầm (delta_max)")

def resource_path(relative_path):
    """Lấy đúng đường dẫn file, kể cả khi chạy .exe"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model = keras.models.load_model(resource_path('fnn_deflection_model.h5'))

# Nhập input
st.subheader("Nhập thông số dầm:")

b = st.number_input("Bề rộng b (m)", min_value=0.01, value=0.15, step=0.01)
h = st.number_input("Chiều cao h (m)", min_value=0.01, value=0.30, step=0.01)
E = st.number_input("Mô đun đàn hồi E (Pa)", min_value=1e5, value=2.0e11, step=1e9, format="%.1e")
L = st.number_input("Chiều dài dầm L (m)", min_value=0.1, value=3.0, step=0.1)
F = st.number_input("Tải trọng F (N)", min_value=0.0, value=5000.0, step=100.0)

if st.button("Dự đoán độ võng cực đại"):
    # Chuẩn bị dữ liệu input
    input_data = np.array([[b, h, E, L, F]])

    # Dự đoán
    delta_max_pred = model.predict(input_data)[0][0]

    st.success(f"✅ Độ võng cực đại dự đoán là: **{delta_max_pred:.6e} m**")

    # Vẽ mô hình cây dầm Cantilever
    x = np.linspace(0, L, 100)
    # Công thức cho dầm cantilever với tải trọng tập trung P ở đầu tự do:
    y = -(F * x**2) / (6 * E * (b * h**3)) * (3 * L - x)  # Độ võng của dầm cantilever

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, y, color='blue', linewidth=3)

    # Trang trí thêm
    ax.set_xlabel("Chiều dài dầm (m)")
    ax.set_ylabel("Độ võng (m)")
    ax.set_title("Mô hình Dầm Cantilever Bị Võng")
    ax.grid(True)
    ax.set_xlim(0, L)
    ax.set_ylim(1.5 * np.min(y), 0.5 * np.max(y))  # chỉnh y để nhìn đẹp

    st.pyplot(fig)
