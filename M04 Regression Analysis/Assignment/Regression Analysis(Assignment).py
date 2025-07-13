import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Part 1. Using the student_performance.csv file, run simple 
# linear regression on the student ‘performance-outcome’ 
# based on ‘hours_studied’ and write 3-4 sentences describing 
# the data results. Include a screenshot of the Python output. 

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

# Part 2. Using the student_performance.csv file, run a multiple linear regression on the 
# student ‘performance-outcome’ based on all the variables in the dataset and write 3-4 sentences 
# describing the data results. Include a screenshot of the Python output.

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

# Part 3 FOR PRACTICE. Using the student_performance.csv file, run a regression 
# model of your choice(Polynomial) that fits continuous data. Challenge yourself to split the 
# data with a 70/30 training/testing spilt and assess the model performance. 
# Write 2-3 sentences describing the data results. Include a screenshot of the Python output.

# Load the data
df = pd.read_csv('student_performance.csv')

# Define the predictor and target variables
target_variable = 'performance_outcome'
predictor_variables = df.drop(columns=[target_variable]).columns

# Create model variables
X = df[predictor_variables]
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the polynomial regression model (sklearn handles intercept automatically)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)

# Make predictions
y_pred = poly_model.predict(X_test)

# Print model information
print("Polynomial Regression Model (Degree 2)")
print("="*50)
print(f"Model Pipeline: {poly_model}")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Get the linear regression component to show coefficients
linear_reg = poly_model.named_steps['linearregression']
poly_features = poly_model.named_steps['polynomialfeatures']

print(f"\nModel Details:")
print(f"Intercept: {linear_reg.intercept_:.4f}")
print(f"Number of features after polynomial transformation: {len(linear_reg.coef_)}")
print(f"Feature names: {poly_features.get_feature_names_out(X.columns)}")

print(f"\nFirst few predictions vs actual:")
for i in range(min(10, len(y_test))):
    print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]:.2f}")
