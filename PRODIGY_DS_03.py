# -*- coding: utf-8 -*-
"""
Created on Sun May 11 23:17:08 2025

@author: jainp
"""

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Load the dataset
df = pd.read_csv("E:/intern/bank+marketing/bank/bank-full.csv", sep=';')  # Make sure you have the correct path

# 3. Inspect the data
print(df.head())
print(df.info())

# 4. Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 5. Define features and target
X = df.drop('y', axis=1)
y = df['y']

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# 9. Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf,
          feature_names=X.columns.tolist(),
          class_names=['no', 'yes'],  # manually provide the class names as a list
          filled=True)



plt.title("Decision Tree for Bank Marketing Prediction")
plt.show()
