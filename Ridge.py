import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

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
features = ['TV', 'Radio', 'Newspaper']
X = data[features]
y = data['Sales']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa các đặc trưng
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tạo mô hình hồi quy Ridge
alpha = 0.01 # Độ mạnh điều chỉnh
ridge_model = Ridge(alpha=alpha)

# Huấn luyện mô hình với dữ liệu huấn luyện
ridge_model.fit(X_train_scaled, y_train)

# Lưu mô hình đã huấn luyện
joblib.dump(ridge_model, 'ridge_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Lưu cả bộ chuẩn hóa nếu cần dự đoán cho dữ liệu mới

# Dự đoán trên tập huấn luyện và kiểm tra
y_train_pred = ridge_model.predict(X_train_scaled)
y_test_pred = ridge_model.predict(X_test_scaled)

# Đánh giá mô hình trên tập huấn luyện
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)

# Đánh giá mô hình trên tập kiểm tra
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

# In kết quả đánh giá
print("Đánh giá mô hình trên tập huấn luyện:")
print(f"R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")

print("\nĐánh giá mô hình trên tập kiểm tra:")
print(f"R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

# Biểu đồ dự đoán trên tập huấn luyện
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='orange', label='Dự đoán (Tập huấn luyện)')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Đường lý tưởng')
plt.xlabel('Doanh số thực tế (Tập huấn luyện)')
plt.ylabel('Doanh số dự đoán (Tập huấn luyện)')
plt.title('Biểu đồ dự đoán Doanh số bằng Ridge Regression (Tập huấn luyện)')
plt.legend()
plt.show()

# Biểu đồ dự đoán trên tập kiểm tra
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Dự đoán (Tập kiểm tra)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Đường lý tưởng')
plt.xlabel('Doanh số thực tế (Tập kiểm tra)')
plt.ylabel('Doanh số dự đoán (Tập kiểm tra)')
plt.title('Biểu đồ dự đoán Doanh số bằng Ridge Regression (Tập kiểm tra)')
plt.legend()
plt.show()

# Tải mô hình đã lưu và chuẩn hóa
ridge_model = joblib.load('ridge_model.pkl')
scaler = joblib.load('scaler.pkl')

# Dự đoán cho dữ liệu mới
new_data = {}
for feature in features:
    value = float(input(f"Nhập giá trị cho {feature} ($): "))
    new_data[feature] = value

# Chuyển đổi dữ liệu mới thành DataFrame và chuẩn hóa
new_data_df = pd.DataFrame([new_data])
new_data_scaled = scaler.transform(new_data_df)

# Dự đoán doanh số cho dữ liệu mới
predicted_sales = ridge_model.predict(new_data_scaled)

# In kết quả dự đoán
print(f"Dự đoán Doanh số: {predicted_sales[0]:.2f}")
