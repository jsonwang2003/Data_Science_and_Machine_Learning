# Part 3. Run a multiple regression to predict the variables that influence quality
# Run a multiple linear regression with a target variable as 'quality' and all the other 
# variables as the predictor variables. A summary of the output with model interpretation.

import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('../winequality.csv')

# Define the target variable (dependent variable)
target_variable = 'quality'

# Define the predictor variables (independent variables)
predictor_variables = df.drop(columns=[target_variable]).columns

# Create the model
X = df[predictor_variables]
y = df[target_variable]

# Add a constant to the predictor variables (intercept term)
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

# Model Interpretation
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

# Example interpretation (you need to adapt this based on your specific variables)
# Iterate through the coefficients and their corresponding features.
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

# Assess the overall model fit based on R-squared, adjusted R-squared, and p-values.
print("The R-squared and Adjusted R-squared values suggest the goodness of fit. Higher values indicate that the model explains a greater proportion of the variance in the target variable.")
print("Examine the p-values for each predictor.  P-values less than your significance level (e.g., 0.05) imply that the corresponding predictor is statistically significant.")
