# install dependencies first:
# pip install pandas numpy matplotlib scikit-learn seaborn xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
# Simulated CSV file: Replace with real traffic/accident dataset
data = pd.read_csv("traffic_data.csv")  

# Sample preprocessing
data.dropna(inplace=True)  # remove missing values
data['weather_encoded'] = data['weather'].astype('category').cat.codes

# Feature selection
features = ['vehicle_count', 'speed_avg', 'weather_encoded', 'time_hour']
X = data[features]
y = data['accident']  # binary: 0 = No accident, 1 = Accident
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Suppose you have lat/lon coordinates for traffic records
plt.scatter(data['longitude'], data['latitude'], c=data['accident'], cmap='coolwarm', alpha=0.5)
plt.title("Accident Risk Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
