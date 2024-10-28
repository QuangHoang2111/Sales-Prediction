import streamlit as st
import joblib
import numpy as np

linear_model = joblib.load('linear_model_sales.pkl')
ridge_model = joblib.load('ridge_model.pkl')
neural_model = joblib.load('neural_model_sales.pkl')
stacking_model = joblib.load('stacking_model.pkl')

scaler = joblib.load('scaler.pkl')

mean_tv = 147.04  
mean_radio = 23.26  
mean_newspaper = 30.55

def predict_sales(model, tv, radio, newspaper):
    input_data = np.array([[tv, radio, newspaper]])
    input_scaled = scaler.transform(input_data)
    predicted_sales = model.predict(input_scaled)
    return predicted_sales[0]

st.title("Dự đoán doanh số quảng cáo")

algorithm = st.selectbox("Chọn thuật toán", ("Linear Regression", "Ridge Regression", "Neural Network", "Stacking"))

tv = st.number_input("Nhập giá trị cho TV (để trống sẽ dùng giá trị trung bình):", value=None)
radio = st.number_input("Nhập giá trị cho Radio (để trống sẽ dùng giá trị trung bình):", value=None)
newspaper = st.number_input("Nhập giá trị cho Newspaper (để trống sẽ dùng giá trị trung bình):", value=None)

if tv is None:
    tv = mean_tv
if radio is None:
    radio = mean_radio
if newspaper is None:
    newspaper = mean_newspaper

if st.button("Dự đoán"):
    if algorithm == "Linear Regression":
        model = linear_model
    elif algorithm == "Ridge Regression":
        model = ridge_model
    elif algorithm == "Neural Network":
        model = neural_model
    elif algorithm == "Stacking":
        model = stacking_model

    predicted_sales = predict_sales(model, tv, radio, newspaper)
    st.success(f'Dự đoán doanh số: {predicted_sales:.2f}')

