# %%
# !pip install mlxtend

import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

#frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)
### alternatively:
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
#frequent_itemsets = fpmax(df, min_support=0.6, use_colnames=True)

frequent_itemsets





# %%

# use supermarket.python.csv instead of supermarket.arff
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth

# load the right file
data_file = 'supermarket.python.csv'
df = pd.read_csv(data_file)

# drop few columns - total isn't needed, is text...
df.drop('total',axis=1,inplace=True)
# TODO: drop few more here department* non-host-support
df.head()

# run apriori
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets
# %%
