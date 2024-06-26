# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:28:40 2024

@author: OMEN
"""
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


diamonds = sns.load_dataset('diamonds')
X = diamonds.drop("cut", axis=1)
y = diamonds['cut']

print(X.shape, y.shape)

print(diamonds.head())



# splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42
)

print(X_train.shape, y_test.shape)


# Define categorical and numerical features
categorical_features = X.select_dtypes(
   include=["object"]
).columns.tolist()

print(f'Categorical Features: {categorical_features}')

numerical_features = X.select_dtypes(
   include=["float64", "int64"]
).columns.tolist()

print(f'Numerical Features: {numerical_features}')

preprocessor = ColumnTransformer(
   transformers=[
       ("cat", OneHotEncoder(), categorical_features),
       ("num", StandardScaler(), numerical_features),
   ]
)


print(f'preprocessor:{preprocessor}')


pipeline = Pipeline(
   [
       ("preprocessor", preprocessor),
       ("classifier", GradientBoostingClassifier(random_state=42)),
   ]
)


print(f'pipeline:{pipeline}')


# Perform 5-fold cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f'cv_scores: {cv_scores}')
# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Generate classification report

report = classification_report(y_test, y_pred)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
print("\nClassification Report:")
print(report)
