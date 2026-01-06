import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


df = pd.read_csv('us_accidents.csv')

df = df[['Severity', 'Start_Time', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
         'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition', 'Sunrise_Sunset']]

df.dropna(inplace=True)

df['Weather_Condition'] = df['Weather_Condition'].astype('category').cat.codes
df['Sunrise_Sunset'] = df['Sunrise_Sunset'].map({'Day': 1, 'Night': 0})

df['Start_Hour'] = pd.to_datetime(df['Start_Time']).dt.hour
df.drop('Start_Time', axis=1, inplace=True)

X = df.drop('Severity', axis=1)
y = df['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'accident_model.pkl')