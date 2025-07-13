# Part 2. Run a simple linear regression to determine to predict quality based on pH level.
# Simple linear regression with a target variable as 'quality' and ‘pH’ as the predictor variable 
# and a summary of the regression output, a chart, and an interpretation of the results
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('../winequality.csv')

# Define the predictor and target variables
X = df['pH']
y = df['quality']

# Add a constant to the predictor variable (intercept term)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())


# Create a scatter plot of the data and the regression line
plt.scatter(df['pH'], df['quality'], label='Data Points')
plt.plot(df['pH'], model.predict(X), color='red', label='Regression Line')
plt.xlabel('pH')
plt.ylabel('Quality')
plt.title('Simple Linear Regression: Quality vs. pH')
plt.legend()
plt.show()

# Interpretation (example)
print("\nInterpretation:")
print("The R-squared value indicates the proportion of variance in 'quality' that is predictable from 'pH'.")
print("The coefficient for 'pH' represents the change in 'quality' for a one-unit increase in 'pH', holding all else constant.")
print("The p-value for 'pH' indicates the statistical significance of the relationship between 'pH' and 'quality'.")
print("The intercept represents the predicted 'quality' when 'pH' is zero.")
print("Further analysis is needed to interpret the model's accuracy and practical implications in the context of wine quality.")
