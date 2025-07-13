#Load the data from go_surfing.csv. Load the data from the specified CSV file into a pandas DataFrame 
import pandas as pd
df = pd.read_csv('../go_surfing.csv')

# Display the first few rows of the DataFrame to confirm the data is loaded
print(df.head())


# Logistic regression with the target variable is 'go_surfing' and the predictor variables are weekend, small_waves, popular_location, good_weather, summer, morning_time, wetsuit_needed, close_drive. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Define predictor and target variables
X = df[['weekend', 'small_waves', 'popular_location', 'good_weather', 'summer', 'morning_time', 'wetsuit_needed', 'close_drive']]
y = df['go_surfing']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression model using statsmodels for summary table
X_train_sm = sm.add_constant(X_train) # Adding a constant for the intercept
model = sm.Logit(y_train, X_train_sm).fit()
print(model.summary())

# Fit logistic regression model using scikit-learn for predictions
clf = LogisticRegression(random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Not Surfing', 'Surfing'],
            yticklabels=['Not Surfing', 'Surfing'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Most predictive variables (based on coefficients)
coefficients = pd.DataFrame({'Variable': X_train.columns, 'Coefficient': clf.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', key=abs, ascending=False)
print("\nMost Predictive Variables:")
print(coefficients)