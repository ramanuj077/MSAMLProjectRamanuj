# MSA ML Project Ramanuj

PROBLEM STATEMENT

Employee attrition is a critical issue for organizations especially in the fast paced industries where retaining talent is essetial.The goal is to predict wheather a employee would leave the companhy or not based on various factors.

WORKFLOW OVERVIEW

1.DATA CLEANING

Handled missing values:
Numerical columns → filled with median
Categorical columns → filled with mode
Dropped irrelevant columns like Over18
FEATURE ENGINEERING
Scaled numerical features using StandardScaler
Encoded categorical features using LabelEncoder
Analyzed feature correlation with Attrition
Model Building
Chose Logistic Regression for its interpretability and baseline performance
Used class_weight='balanced' to handle class imbalance
Split data into 80% training and 20% testing
Evaluation
Metrics used: Accuracy, Precision, Recall, F1-score
Visualized performance using a confusion matrix and count plots
RESULTS

Accuracy: 71.7%
Precision (Class 1): 26.9%
Recall (Class 1): 61.5% Confusion Matrix:
True Positives: 24
False Positives: 68
False Negatives: 15
True Negatives: 187
The result code is in the code.ipynb(last part)
