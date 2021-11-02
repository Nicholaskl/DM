# %%
# !pip install mlxtend

# This is an example task for the basics :)

import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# Turn into numpy bool array with columns labeled as above
# True if an item exists in a set, otherwise False
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
# Crete the dataframe with those given column names and values
df = pd.DataFrame(te_ary, columns=te.columns_)

print("== fpgrowth ==")
growth_frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)
print(growth_frequent_itemsets)

print("== apriori ==")
apriori_frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print(apriori_frequent_itemsets)

print("== fpmax ==")
max_frequent_itemsets = fpmax(df, min_support=0.6, use_colnames=True)
print(max_frequent_itemsets)

# %%

# use supermarket.python.csv instead of supermarket.arff

# load the right file
data_file = 'supermarket.python.csv'
df = pd.read_csv(data_file)

# drop few columns - total isn't needed, is text...
df.drop('total',axis=1,inplace=True)
# non host support has spaces so have to do this weird thing
df.drop("'non host support'",axis=1,inplace=True)
# get rid of all the department columns
df.drop(list(df.filter(regex='department')), axis=1, inplace=True)

# run apriori
supermarket_frequent = apriori(df, min_support=0.2, use_colnames=True)
# supermarket_frequent = fpgrowth(df, min_support=0.2, use_colnames=True)

rules = association_rules(supermarket_frequent,min_threshold=0.8, metric='confidence')

# filter the rules and sort according to confidence level
rules = rules[['antecedents', 'consequents', 'confidence']]
rules = rules[rules['consequents']==frozenset({"'bread and cake'"})]
rules.sort_values(by=['confidence'],axis=0,ascending=False, inplace=True)
# %%
