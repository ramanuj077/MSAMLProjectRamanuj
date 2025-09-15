import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

employee_status=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
employee_status.head()
data=employee_status.copy()
for col in data.columns:
    if data[col].dtype in ['float64', 'int64']:
        data[col] = data[col].fillna(data[col].median())
    else:
        data[col] = data[col].fillna(data[col].mode()[0])

data.drop(columns=['Over18'])

data.to_csv("clean_employee_attrition.csv", index=False)
from sklearn.preprocessing import StandardScaler

num_cols = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
            "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
            "YearsSinceLastPromotion"]

scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import seaborn as sb
for i in data.select_dtypes(include=['object']).columns:
    data[i] = le.fit_transform(data[i])
corellation = data.corr()['Attrition'].sort_values(ascending=False)
sb.heatmap(corellation.to_frame(), annot=False, cmap='coolwarm')
import matplotlib.pyplot as plt
sb.countplot(x='Attrition', data=data)
plt.title('Employee Attrition Count')
plt.show()

features=data.drop('Attrition', axis=1)
target=data['Attrition']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
import pickle
with open('employee_attrition_model.pkl', 'wb') as file:
    pickle.dump(model, file)
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

y_pred = model.predict(X_test)




