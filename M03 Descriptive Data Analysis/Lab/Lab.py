# Part 1. Load the data_science_salaries.csv file into the Python environment
# load data_science_salaries.csv
import pandas as pd
import matplotlib.pyplot as plt

# load the dataframe.
df = pd.read_csv('../data_science_salaries.csv')

# print information
df.info()
df.head()

# Part 2. Calculate counts of experience level 
# group data by 'experience_level’ and count occurrences
experience_level_counts = df.groupby('experience_level')['experience_level'].count()
print("Experience Level Counts:\n", experience_level_counts)

# Part 3. Calculate descriptive statistics for salary
# descriptive statistics for ‘salary’
print(df['salary'].describe())

# Part 4. Calculate the average salary by experience level
# group data by ‘experience_level’ and calculate the average ‘salary’ for each group
average_salary_by_experience = df.groupby('experience_level')['salary'].mean()

# print the result
print(average_salary_by_experience)

# Part 5. Create a formatted table by employment types and company size
# create a formatted table based on counts of 'employment_type' and 'company_size'
# group data by 'employment_type' and 'company_size', then count occurrences
grouped_data = df.groupby(['employment_type', 'company_size']).size().unstack(fill_value=0)

# display the formatted table
print("Counts of Employment Type and Company Size:\n")
print(grouped_data)

# Part 6. Create a bar chart by experience level counts
# create a bar chart based on the counts of 'experience_level'
# create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(experience_level_counts.index, experience_level_counts.values)
plt.xlabel("Experience Level")
plt.ylabel("Number of Employees")
plt.title("Distribution of Experience Levels")
plt.show()

# Part 7. Create a pie chart by employment type counts. 
# create a pie chart based on the counts of 'employment_type'
# create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(experience_level_counts.values, labels=experience_level_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Experience Levels")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()