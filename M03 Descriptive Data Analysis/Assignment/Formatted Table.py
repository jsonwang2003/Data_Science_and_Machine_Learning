# Part 3. Using the data_science_salaries.csv file, create a table 
# that is the average salary by experience level and include the 
# data summary in a formatted table. Write 1-2 sentences describing 
# the data results.  
import pandas as pd

df = pd.read_csv('../data_science_salaries.csv')

# create a formatted table based on counts of 'employment_type' and 'company_size'
# group data by 'employment_type' and 'company_size', then count occurrences
grouped_data = df.groupby(['experience_level', 'salary']).size().unstack(fill_value=0)

# display the formatted table
print("Counts of Employment Type and Company Size:\n")
print(grouped_data)