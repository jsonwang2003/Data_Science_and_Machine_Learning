# Part 2. Using the student_performance.csv file, run a multiple linear regression on the 
# student ‘performance-outcome’ based on all the variables in the dataset and write 3-4 sentences 
# describing the data results. Include a screenshot of the Python output.

import pandas as pd
import statsmodels.api as sm

# Load the data
df = pd.read_csv('student_performance.csv')

# Define the predictor and target variables
target_variable = 'performance_outcome'
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