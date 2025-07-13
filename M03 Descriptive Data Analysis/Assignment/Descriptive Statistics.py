# Part 1. Using the data_science_salaries.csv file, run descriptive statistics on the data 
# science salary data by experience level and write 3-4 sentences describing the data results. 
# Include a screenshot of the Python output. 
import pandas as pd

# Load the dataset
df = pd.read_csv('../data_science_salaries.csv')

# group data by 'experience_level’ and count occurrences
experience_level_counts = df.groupby('experience_level')['experience_level'].count()
print("Experience Level Counts:\n", experience_level_counts)

# descriptive statistics for 'salary'
print(df['experience_level'].describe())

# group data by 'experience_level' and calculate average 'salary' for each group
average_salary_by_experience = df.groupby('experience_level')['salary'].mean()
print("Average Salary by Experience Level:\n", average_salary_by_experience)