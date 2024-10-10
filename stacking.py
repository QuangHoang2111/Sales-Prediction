import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Bước 1: Đọc dữ liệu
data = pd.read_csv('advertisite.csv')

# Tiền xử lý dữ liệu
# Xóa các dữ liệu bị trống cho các cột TV, Radio, Newspaper, và Sales
data.dropna(subset=['TV', 'Radio', 'Newspaper', 'Sales'], inplace=True)

# Sửa lại các dữ liệu âm cho các cột TV, Radio, Newspaper, và Sales
for x in data.index:
    if data.loc[x, "TV"] < 0:
        data.loc[x, "TV"] = np.abs(data.loc[x, "TV"])
    if data.loc[x, "Radio"] < 0:
        data.loc[x, "Radio"] = np.abs(data.loc[x, "Radio"])
    if data.loc[x, "Newspaper"] < 0:
        data.loc[x, "Newspaper"] = np.abs(data.loc[x, "Newspaper"])
    if data.loc[x, "Sales"] < 0:
        data.loc[x, "Sales"] = np.abs(data.loc[x, "Sales"])   

# Xóa dữ liệu trùng lặp
data.drop_duplicates(inplace=True)

# Giả sử dữ liệu có các cột 'TV', 'Radio', 'Newspaper', 'Sales'
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Bước 2: Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 3: Tải các mô hình đã lưu
linear_model = joblib.load('linear_model_sales.pkl')
ridge_model = joblib.load('ridge_model.pkl')
neural_model = joblib.load('neural_model_sales.pkl')

# Bước 4: Tải scaler nếu cần thiết
scaler = joblib.load('scaler.pkl')

# Chuẩn hóa dữ liệu
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bước 5: Tạo StackingRegressor với các mô hình đã tải
stacking_model = StackingRegressor(
    estimators=[
        ('linear', linear_model),
        ('ridge', ridge_model),
        ('neural', neural_model)
    ],
    final_estimator=Ridge()
)

# Huấn luyện mô hình stacking
stacking_model.fit(X_train_scaled, y_train)

# Bước 6: Lưu mô hình Stacking
joblib.dump(stacking_model, 'stacking_model.pkl')
print("Mô hình Stacking đã được lưu.")

# Bước 7: Dự đoán trên tập train và test
y_train_pred = stacking_model.predict(X_train_scaled)
y_test_pred = stacking_model.predict(X_test_scaled)

# Bước 8: Đánh giá mô hình
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("Đánh giá mô hình trên tập huấn luyện:")
print(f"R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")

print("\nĐánh giá mô hình trên tập kiểm tra:")
print(f"R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

# Bước 9: Vẽ biểu đồ so sánh giữa giá trị thực và giá trị dự đoán cho cả tập train và test
plt.figure(figsize=(14, 6))

# Biểu đồ cho tập train
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Biểu đồ dự đoán Doanh số bằng Stacking (Tập huấn luyện)')
plt.xlabel('Giá trị thực')
plt.ylabel('Giá trị dự đoán')

# Biểu đồ cho tập test
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='orange', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Biểu đồ dự đoán Doanh số bằng Stacking (Tập kiểm tra)')
plt.xlabel('Giá trị thực')
plt.ylabel('Giá trị dự đoán')

plt.tight_layout()
plt.show()

# Bước 10: Nhập dữ liệu để dự đoán Sales
def predict_sales(tv, radio, newspaper):
    input_data = np.array([[tv, radio, newspaper]])
    input_scaled = scaler.transform(input_data)
    predicted_sales = stacking_model.predict(input_scaled)
    return predicted_sales[0]

# Nhập dữ liệu để dự đoán
tv_input = float(input("Nhập giá trị cho TV: "))
radio_input = float(input("Nhập giá trị cho Radio: "))
newspaper_input = float(input("Nhập giá trị cho Newspaper: "))

predicted_sales = predict_sales(tv_input, radio_input, newspaper_input)
print(f'Dự đoán doanh số: {predicted_sales:.2f}')
