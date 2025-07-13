# Part 1. Load the winequality.csv file into the Python environment 
# Load the data from winequality.csv
import pandas as pd

# Load the data from the specified CSV file into a pandas DataFrame
df = pd.read_csv('/content/winequality.csv')

# Display the first few rows of the DataFrame to confirm the data is loaded
print(df.head())
