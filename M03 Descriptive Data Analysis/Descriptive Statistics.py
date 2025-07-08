import pandas as pd
import matplotlib.pyplot as plt

# load data_science_salaries.csv
# load the dataframe.
df = pd.read_csv('/content/data_science_salaries.csv')

# print information
df.info()
df.head()

# group data by 'experience_level’ and count occurrences
experience_level_counts = df.groupby('experience_level')['experience_level'].count()
print("Experience Level Counts:\n", experience_level_counts)

# descriptive statistics for ‘salary’
print(df['salary'].describe())

# group data by ‘experience_level’ and calculate the average ‘salary’ for each group
average_salary_by_experience = df.groupby('experience_level')['salary'].mean()
# print the result
print(average_salary_by_experience)

# create a formatted table based on counts of 'employment_type' and 'company_size'
# group data by 'employment_type' and 'company_size', then count occurrences
grouped_data = df.groupby(['employment_type', 'company_size']).size().unstack(fill_value=0)

# display the formatted table
print("Counts of Employment Type and Company Size:\n")
print(grouped_data)

# create a bar chart based on the counts of 'experience_level'
# create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(experience_level_counts.index, experience_level_counts.values)
plt.xlabel("Experience Level")
plt.ylabel("Number of Employees")
plt.title("Distribution of Experience Levels")
plt.show()

# create a pie chart based on the counts of 'employment_type'
# create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(experience_level_counts.values, labels=experience_level_counts.index, autopct='%1.1f%%',
startangle=90)
plt.title("Distribution of Experience Levels")
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()