from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets

# descriptive statistics
print(X.describe())
print(y.value_counts())

# visualize feature distributions
X.hist(bins=30, figsize=(15, 10))
plt.suptitle('Feature Distributions')
plt.show()