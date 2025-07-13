# Part 5. Run a Gradient Boosting Regression to predict quality based on all other variables
# Gradient Boosting Regression with a target variable as 'quality' and all the other variables as the predictor variables. A summary of the output with model interpretation and a tree

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../winequality.csv')

# Define features (X) and target (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Initialize and fit the Gradient Boosting Regressor
gbr = GradientBoostingRegressor()  # Use default parameters for now
gbr.fit(X, y)

#Feature Importance
feature_importance = gbr.feature_importances_
feature_names = X.columns
for i, importance in enumerate(feature_importance):
    print(f"Feature: {feature_names[i]}, Importance: {importance:.4f}")

# Model Interpretation Summary (based on feature importance)
print("\nModel Interpretation:")
print("The Gradient Boosting Regressor identifies the following as the most important features for predicting wine quality:")
# Print the top 3 most important features
sorted_feature_importance_indices = feature_importance.argsort()[::-1]
for i in sorted_feature_importance_indices[:3]:
  print(f"- {feature_names[i]} (Importance: {feature_importance[i]:.4f})")
print("Other features also contribute, but to a lesser extent.")
