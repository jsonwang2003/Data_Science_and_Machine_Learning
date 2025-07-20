# Using the Market_Basket.csv file, run an Apriori Algorithm 

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load dataset
market_df = pd.read_csv('../Market_Basket.csv')

# Preprocess and group transactions
transactions = market_df.apply(lambda row: [item for item in row if pd.notnull(item)], axis=1).tolist()

# Encode transactions
te = TransactionEncoder()
transactions_encoded = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# Adjust min_support based on the item frequency (set it lower, e.g., 0.005)
frequent_itemsets = apriori(transactions_encoded, min_support=0.005, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Sort and display top 10 rules by lift
rules_sorted = rules.sort_values(by='lift', ascending=False).head(10)
print(rules_sorted)
