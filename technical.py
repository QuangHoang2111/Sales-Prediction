import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("advertisite.csv")

def categorize_sales(sales):
    if sales < 10:
        return 'Thấp'
    elif 10 <= sales < 15:
        return 'Trung bình'
    else:
        return 'Cao'

df['Sales_Category'] = df['Sales'].apply(categorize_sales)

label_encoder = LabelEncoder()
df['Sales_Category'] = label_encoder.fit_transform(df['Sales_Category'])

X = df[['TV', 'Newspaper', 'Radio']]
y = df['Sales_Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=0.1),
    "Neural": MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42),
}

estimators = [('lr', models["Linear"]), ('ridge', models["Ridge"]), ('neural', models["Neural"])]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

models["Stacking"] = stacking_model

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

 
    y_pred = model.predict(X_test_scaled)
    y_pred = np.round(y_pred).astype(int) 
    y_pred_labels = label_encoder.inverse_transform(y_pred)  
    y_test_labels = label_encoder.inverse_transform(y_test)  


    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=["Thấp", "Trung bình", "Cao"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Thấp", "Trung bình", "Cao"], yticklabels=["Thấp", "Trung bình", "Cao"])
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(f"Ma trận nhầm lẫn - {name}")
    plt.show()
