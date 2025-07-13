# Part 2. Using the data_science_salaries.csv file, run counts on 
# employment type and company size and report the data in a bar 
# graph. Write 1-2 sentences describing the data results.
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../data_science_salaries.csv')

# create a formatted table based on counts of 'experience_level' and 'company_size'
grouped_data = df.groupby(['employment_type', 'company_size']).size().unstack(fill_value=0)

# create a bar chart based on the counts of 'experience_level'
# create the bar chart
plt.figure(figsize=(10, 6))
grouped_data.plot(kind='bar', stacked=True, ax=plt.gca())
plt.xlabel("Emplopyment Type")
plt.ylabel("Number of Employees")
plt.title("Distribution of Employment Type verses Company Size")
plt.tight_layout()
plt.show()