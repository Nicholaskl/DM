#%%
from enum import unique
from numpy.lib.function_base import select
import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt


data_file = 'kddcup.data_10_percent_corrected_10000_samples.arff'
data, meta = arff.loadarff(data_file)
df=pd.DataFrame(data=data)



## Only keep nomial attributes

df = df.select_dtypes(include='object')



## Remove duplicate rows

duplicates = df.duplicated(subset=None, keep='first')
duplicate_rows = np.where(duplicates == True)[0]

df.drop(index=duplicate_rows, axis=0, inplace=True)

rows = df.shape[0]

## Get a list of all nominal values for each attribute
unique_vals = []

# Loop over all the attributes and add all unique ones to a list
for attribute in df:
    unique_vals.append(df.loc[:,attribute].unique())

attribute_names = df.columns.values.tolist()

## Select a random row - use shape[0] for number of rows
selected_row = np.random.randint(0,df.shape[0])    

# Use selected_row = 86 to match Sonny's test

x = df.iloc[selected_row,:].values
cols = x.shape[0]


## Calculate the weight
prob_weights = np.zeros((cols,), dtype=float) # initialise array
value_counts = []

for col in range(cols):
    nominal_value = x[col]

    # Find where in the unique values list the value is
    loc = np.where(unique_vals[col]==nominal_value) 

    # Count occurences of nominal value 
    counts = df[attribute_names[col]].value_counts().values
    value_counts.append(counts[loc])

for col in range(cols):
    symbol_count = value_counts[col]
    prob_symbol = symbol_count/rows
    prob_weights[col] = (1/(prob_symbol**2)) 
    # So this gives us S(xi, yi)


## Compute overlap similarity scores

similarity_scores = []
for row in range(rows):
    if (row != selected_row):
        currRow = df.iloc[row,:].values

        same_cols = np.where(currRow== x)[0]

        similarity = 0
        for same_col in same_cols:
            similarity += prob_weights[same_col]
            
        similarity_scores.append(similarity)
    else:
        similarity_scores.append(0)

# np.array(similarity_scores)


# %%

# hint 1: using numpy.argmax() to find the index of the max value
nearest = np.argmax(similarity_scores)
plt.plot(range(rows),similarity_scores)
plt.ylabel('similarity')
plt.xlabel('rows')
plt.title('similarity with inverse freq')
plt.show() 

print("nearest sample = ",nearest)


# %%
import arff

arff.dump('test.arff'
      , df.values
      , relation='relation name'
      , names=df.columns)