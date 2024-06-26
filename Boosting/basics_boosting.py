# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:13:14 2024

@author: Sajjan 
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging 
import mlflow 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import classification_report


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

mlflow.autolog()

# to visualize the ml flow after training
# mlflow ui --port 8080

# random data 
X, y = make_classification(n_samples=5000, n_features=200, n_informative=20, n_redundant=15, n_clusters_per_class=6, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")


ada = AdaBoostClassifier(n_estimators=100, random_state=42)

ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
ada_accuracy = accuracy_score(y_test, y_pred_ada)
print(f"Ada Boosting Accuracy: {ada_accuracy:.2f}")


print("AdaBoost Classification Report:")
print(classification_report(y_test, y_pred_ada))

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
