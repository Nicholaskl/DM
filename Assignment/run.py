#%%
import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt

df=pd.read_csv('data2021.student.csv', sep=',')

# duplicates = df.duplicated(subset=None, keep='first')
# duplicate_rows = np.where(duplicates == True)[0]

# df.drop(index=duplicate_rows, axis=0, inplace=True)

rows = df.shape[0]

# Find and remove attributes with more than 80% missing values

# Will find the missing entries in columns
# Prints out attribute name and how many are missings
# Imports: df (DataFrame)
# Exports: None 
def missingEntries(df):
    attribute_names = df.columns
    num_missing = []
    for attr in attribute_names:
        sum = df[attr].isnull().sum()
        if sum > 1:
            num_missing.append(attr + " " + str(sum)) # Gets sum of missing elements in column

    print("Missing numbers are")
    print(num_missing)

def duplicates(df):

    # Find duplicate instances
    duplicates = df.duplicated(subset=None, keep='first')
    duplicate_rows = np.where(duplicates == True)[0]

    for duplicate_row in duplicate_rows:
        print("Dropped row", duplicate_row)
        # df.drop(index=duplicate_row, axis=0, inplace=True)

    # Find duplicate attributes
    duplicate_cols = []
    for col in range(df.shape[1]): # loop through columns
        curr_col = df.iloc[:,col]

        # Now loop through each subsequent column
        for checkCol in range(col+1, df.shape[1]):
            check_col = df.iloc[:, checkCol]

            # Check if the columns match
            # Have to do this strange equality on arrays
            if (curr_col.values == check_col.values).all():
                # Get the name of the attribute column
                duplicate_cols.append(curr_col.name)
    print(duplicate_cols)


# %%
