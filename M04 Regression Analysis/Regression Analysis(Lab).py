import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from winequality.csv
# Load the data from the specified CSV file into a pandas DataFrame
df = pd.read_csv('winequality.csv')

# Display the first few rows of the DataFrame to confirm the data is loaded
print(df.head())

# Simple linear regression with a target variable as 'quality' and 'pH' as the 
# predictor variable and a summary of the regression output, a chart, and an 
# interpretation of the results

# # Define the predictor and target variables
X = df['pH']
y = df['quality']

# # Add a constant to the predictor variable (intercept term)
X = sm.add_constant(X)

# # Fit the linear regression model
model = sm.OLS(y, X).fit()

# # print the regression summary
print(model.summary())

# # Create a scatter plot of the data and the regression line
plt.scatter(df['pH'], df['quality'], label='Data Points')
plt.plot(df['pH'], model.predict(X), color='red', label='Regression Line')
plt.xlabel('pH')
plt.ylabel('Quality')
plt.title('Simple Linear Regression: Quality vs. pH')
plt.legend()
plt.show()

# # Interpretation (example)
print("\nInterpretation:")
print("The R-squared value indicates the proportion of variance in 'quality' that is predictable from 'pH'.")
print("The coefficient for 'pH' represents the change in 'quality' for a one-unit increase in 'pH', holding all else constant.")
print("The p-value for 'pH' indicates the statistical significance of the relationship between 'pH' and 'quality'.")
print("The intercept represents the predicted 'quality' when 'pH' is zero.")
print("Further analysis is needed to interpret the model's accuracy and practical implications in the context of wine quality.")

# Run a multiple linear regression with a target variable as 'quality' and all the other variables as the predictor variables. A summary of the output with model interpretation.
# Define the target variable (dependent variable)
target_variable = 'quality'

# # Define the predictor variables (independent variables)
predictor_variables = df.drop(columns=[target_variable]).columns

# # Create the model
X = df[predictor_variables]
y = df[target_variable]

# # Add a constant to the predictor variables (intercept term)
X = sm.add_constant(X)

# # Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# # Print the model summary
print(model.summary())

# # Model Interpretation
print("\nModel Interpretation:")
print("R-squared:", model.rsquared)
print("Adjusted R-squared:", model.rsquared_adj)

print("\nCoefficients:")
for col, coef in zip(X.columns, model.params):
    print(f"{col}: {coef}")

print("\nP-values:")
for col, p_value in zip(X.columns, model.pvalues):
    print(f"{col}: {p_value}")

print("\nInterpretation of Coefficients:")

# # Example interpretation (you need to adapt this based on your specific variables)
# # Iterate through the coefficients and their corresponding features.
for i in range(len(model.params)):
  if i==0:
    print(f"The intercept is {model.params[i]}.")
    continue

  feature = X.columns[i]
  coefficient = model.params[i]
  p_value = model.pvalues[i]

  print(f"A one unit increase in '{feature}' is associated with a {coefficient:.2f} change in 'quality'.")
  if p_value < 0.05:
    print("This relationship is statistically significant (p < 0.05).")
  else:
    print("This relationship is not statistically significant (p >= 0.05).")
  print("-" * 20)

print("\nOverall Model Assessment:")

# # Assess the overall model fit based on R-squared, adjusted R-squared, and p-values.
print("The R-squared and Adjusted R-squared values suggest the goodness of fit. Higher values indicate that the model explains a greater proportion of the variance in the target variable.")
print("Examine the p-values for each predictor.  P-values less than your significance level (e.g., 0.05) imply that the corresponding predictor is statistically significant.")

# Gradient Boosting Regression with a target variable as 'quality' and all the other variables as the predictor variables. A summary of the output with model interpretation and a tree

df = pd.read_csv('/content/winequality.csv')

# Define features (X) and target (y)
X = df.drop('quality', axis=1)
y = df['quality']

# # Initialize and fit the Gradient Boosting Regressor
gbr = GradientBoostingRegressor()  # Use default parameters for now
gbr.fit(X, y)

# #Feature Importance
feature_importance = gbr.feature_importances_
feature_names = X.columns
for i, importance in enumerate(feature_importance):
    print(f"Feature: {feature_names[i]}, Importance: {importance:.4f}")

# # Model Interpretation Summary (based on feature importance)
print("\nModel Interpretation:")
print("The Gradient Boosting Regressor identifies the following as the most important features for predicting wine quality:")
# Print the top 3 most important features
sorted_feature_importance_indices = feature_importance.argsort()[::-1]
for i in sorted_feature_importance_indices[:3]:
  print(f"- {feature_names[i]} (Importance: {feature_importance[i]:.4f})")
print("Other features also contribute, but to a lesser extent.")

# Gradient Boosting Regression with a target variable as 'quality' and all the other variables as the predictor variables. A summary of the output with model interpretation.

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
