#%%
import pandas as pd
import numpy as np
from pandas.core.arrays.categorical import contains

## Read the csv file. Can do `index_col=8` to pick important column
df = pd.read_csv('D:\CURTIN\YEAR3\Sem2\DM\p1\data.csv', sep=',')

## To only show the top 10 rows
pd.set_option('display.max_columns', 500)
df.head(10)

## Show number of attributes and instances
print("Numer of instances: ", df.shape[0])
print("Numer of attributes: ", df.shape[1])
#%%

# Print out current data type info
df.info()

# for some reason this needs a numpy number? data type probs
# This is if you want to look at numerical values
df.describe(include=[np.number])
#%%


# This is to look at nominal values
df.describe(include=[object])
#%%

## Histogram against survived
df.groupby("survived").hist()

#%%

# Find all the missing values
attribute_names = df.columns

for attr in attribute_names:
    miss_vals = df.loc[:,attr].isnull()
    miss_instances = np.array(np.where(miss_vals==True))
    if miss_instances.size>0:
        print(miss_instances.size, "missing instances of attribute: ",attr)
        print(miss_instances)


# for obj in file.columns:
#     print('hi')

# 

# %%
 