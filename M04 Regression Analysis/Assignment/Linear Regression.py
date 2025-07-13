# Part 1. Using the student_performance.csv file, run simple 
# linear regression on the student ‘performance-outcome’ 
# based on ‘hours_studied’ and write 3-4 sentences describing 
# the data results. Include a screenshot of the Python output. 

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# load the data from student_performance.csv
df = pd.read_csv('../student_performance.csv')

# Define the predictor and target variables
X = df['hours_studied']
y = df['performance_outcome']

# Add a constant to the predictor variable (intercept term)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())

# Create a scatter plot of the data and the regression line
plt.scatter(df['hours_studied'], df['performance_outcome'], label='Data Points')
plt.plot(df['hours_studied'], model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Outcome')
plt.legend()
plt.show()