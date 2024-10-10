import streamlit as st
import joblib
import numpy as np

# Tải các mô hình đã huấn luyện
linear_model = joblib.load('linear_model_sales.pkl')
ridge_model = joblib.load('ridge_model.pkl')
neural_model = joblib.load('neural_model_sales.pkl')
stacking_model = joblib.load('stacking_model.pkl')

# Tải scaler đã sử dụng để chuẩn hóa dữ liệu
scaler = joblib.load('scaler.pkl')

# Hàm dự đoán doanh số
def predict_sales(model, tv, radio, newspaper):
    input_data = np.array([[tv, radio, newspaper]])
    input_scaled = scaler.transform(input_data)
    predicted_sales = model.predict(input_scaled)
    return predicted_sales[0]

# Giao diện Streamlit
st.title("Dự đoán doanh số quảng cáo")

# Lựa chọn thuật toán
algorithm = st.selectbox("Chọn thuật toán", ("Linear Regression", "Ridge Regression", "Neural Network", "Stacking"))

# Nhập thông tin TV, Radio, Newspaper
tv = st.number_input("Nhập giá trị cho TV:", min_value=0.0, value=100.0)
radio = st.number_input("Nhập giá trị cho Radio:", min_value=0.0, value=50.0)
newspaper = st.number_input("Nhập giá trị cho Newspaper:", min_value=0.0, value=30.0)

# Nút để dự đoán
if st.button("Dự đoán"):
    if algorithm == "Linear Regression":
        model = linear_model
    elif algorithm == "Ridge Regression":
        model = ridge_model
    elif algorithm == "Neural Network":
        model = neural_model
    elif algorithm == "Stacking":
        model = stacking_model

    # Dự đoán và hiển thị kết quả
    predicted_sales = predict_sales(model, tv, radio, newspaper)
    st.success(f'Dự đoán doanh số: {predicted_sales:.2f}')

# Chạy ứng dụng bằng lệnh dưới đây trong terminal
# streamlit run your_script_name.py
