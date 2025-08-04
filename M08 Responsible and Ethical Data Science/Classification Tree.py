from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Classification Decision Tree
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# make predictions
y_pred = clf.predict(X_test)

# evaluate classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# visualize decision tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=np.unique(y), filled=True)
class_names = np.unique(y)
plt.title("Decision Tree for Predicting Student Outcomes")
plt.show()
