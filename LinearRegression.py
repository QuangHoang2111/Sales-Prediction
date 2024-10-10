import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('advertisite.csv')

# Tiền xử lý dữ liệu

# Xóa các dữ liệu bị trống cho các cột TV, Radio, Newspaper, và Sales
data.dropna(subset=['TV', 'Radio', 'Newspaper', 'Sales'], inplace=True)

# Sửa lại các dữ liệu âm cho các cột TV, Radio, Newspaper, và Sales
for x in data.index:
    if data.loc[x, "TV"] < 0:
        data.loc[x, "TV"] = np.abs(data.loc[x, "TV"])
        
for x in data.index:
    if data.loc[x, "Radio"] < 0:
        data.loc[x, "Radio"] = np.abs(data.loc[x, "Radio"])

for x in data.index:
    if data.loc[x, "Newspaper"] < 0:
        data.loc[x, "Newspaper"] = np.abs(data.loc[x, "Newspaper"])

for x in data.index:
    if data.loc[x, "Sales"] < 0:
        data.loc[x, "Sales"] = np.abs(data.loc[x, "Sales"])   

# Xóa dữ liệu trùng lặp
data.drop_duplicates(inplace=True)

# Sử dụng các cột cần thiết và mục tiêu là Sales
features = ['TV', 'Radio', 'Newspaper']  # Cột đặc trưng
target = 'Sales'  # Cột mục tiêu

# Đặt các đặc trưng (X) và mục tiêu dự đoán
X = data[features]
y = data[target]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình Linear Regression
linear_model = LinearRegression()

# Huấn luyện mô hình
linear_model.fit(X_train, y_train)

# Lưu mô hình đã huấn luyện
joblib.dump(linear_model, 'linear_model_sales.pkl')
print("Mô hình đã được lưu vào file 'linear_model_sales.pkl'.")

# Tải lại mô hình từ file đã lưu
loaded_model = joblib.load('linear_model_sales.pkl')
print("Mô hình đã được tải lại từ file.")

# Dự đoán trên tập huấn luyện và tập kiểm tra
y_train_pred = loaded_model.predict(X_train)
y_test_pred = loaded_model.predict(X_test)

# Đánh giá mô hình
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# Đánh giá trên tập huấn luyện và kiểm tra
r2_train, rmse_train, mae_train = evaluate_model(y_train, y_train_pred)
r2_test, rmse_test, mae_test = evaluate_model(y_test, y_test_pred)

# In kết quả đánh giá
print("Đánh giá trên tập huấn luyện:")
print(f"R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
print("\nĐánh giá trên tập kiểm tra:")
print(f"R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

# Biểu đồ cho tập huấn luyện
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='orange', label='Dự đoán (Tập huấn luyện)')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Đường lý tưởng')
plt.xlabel('Doanh số thực tế (Tập huấn luyện)')
plt.ylabel('Doanh số dự đoán (Tập huấn luyện)')
plt.title('Biểu đồ dự đoán doanh số bằng Linear Regression (Tập huấn luyện)')
plt.legend()
plt.show()

# Biểu đồ cho tập kiểm tra
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Dự đoán (Tập kiểm tra)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Đường lý tưởng')
plt.xlabel('Doanh số thực tế (Tập kiểm tra)')
plt.ylabel('Doanh số dự đoán (Tập kiểm tra)')
plt.title('Biểu đồ dự đoán doanh số bằng Linear Regression (Tập kiểm tra)')
plt.legend()
plt.show()

# Nhập dữ liệu mới từ người dùng
new_data = {
    'TV': float(input("Nhập giá trị cho TV: ")),
    'Radio': float(input("Nhập giá trị cho Radio: ")),
    'Newspaper': float(input("Nhập giá trị cho Newspaper: "))
}

# Chuyển đổi dữ liệu mới thành DataFrame và chuẩn hóa
new_data_df = pd.DataFrame([new_data])
new_data_scaled = scaler.transform(new_data_df[features])

# Dự đoán doanh số
predicted_sales = loaded_model.predict(new_data_scaled)

# In kết quả dự đoán
print(f"Dự đoán doanh số: {predicted_sales[0]:.2f}")
