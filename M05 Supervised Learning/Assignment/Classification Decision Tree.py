# Using the shopping.csv file, run a logistic regression OR 
# a classification tree to determine  ‘purchase’ based on 
# the seven other variables

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn import tree
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('../shopping.csv')

# Define predictor and target variables
X = df[['holiday_time', 'discount', 'available_in_store', 'late_night', 'good_weather', 'free_delivery', 'limited_supply']]
y = df['purchase']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limiting depth for visualization
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Create a summary table
summary_table = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})
summary_table = summary_table.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(summary_table)

# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Not Purchased', 'Purchased'], filled=True, rounded=True, fontsize=10)
plt.show()