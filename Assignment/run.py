#%%
import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt

df=pd.read_csv('data2021.student.csv', sep=',')

rows = df.shape[0]

def preProcessing(df):
    removeIrrelevant(df, remove=True)
    missingEntries(df, remove=True)
    duplicates(df, remove=True)
    attributeTypes(df)


def removeIrrelevant(df, remove=False):
    # ID is not needed at all
    df.drop(['ID'], axis=1, inplace=True)

    ## Will check if cols contain 1 value
    # The number of unique values in the column
    unique = df.nunique (axis=0)
    for index, value in unique.items():
        if value == 1:
            if remove:
                df.drop(index, axis=1, inplace=True)
            else:
                print(f"{index} only has {value} values")


# Will find the missing entries in columns
# Prints out attribute name and how many are missings
# Imports: df (DataFrame)
# Exports: None 
def missingEntries(df, remove=False):
    attribute_names = df.columns
    num_missing = []
    for attr in attribute_names:
        sum = df[attr].isnull().sum()
        if sum > 1:
            # Gets sum of missing elements in column
            if remove:
                # Remove entries with more than 80% missing values
                # But don't remove the class column
                if (sum/df.shape[1] > 0.8) and (attr != 'Class'):
                    df.drop(attr, axis=1, inplace=True)
                elif (sum/df.shape[1] > 0) and (attr != 'Class'):
                    dt = df[attr].dtype
                    if(dt == np.int64 or dt == np.float64):
                        # in place means don't have to set the variable
                        df[attr].fillna(df[attr].mean(), inplace=True)
                    elif (df[attr].dtype == object):
                        df[attr].fillna(df[attr].mode(), inplace=True)
            else:
                print(attr + " " + str(sum))

def duplicates(df, remove=False):

    # Find duplicate instances
    duplicates = df.duplicated(subset=None, keep='first')
    duplicate_rows = np.where(duplicates == True)[0]

    for duplicate_row in duplicate_rows:
        if remove:
            df.drop(index=duplicate_row, axis=0, inplace=True)
        else:
            print("Duplicate row", duplicate_row)
        
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
                duplicate_cols.append([curr_col.name, check_col.name])
    for col in duplicate_cols:
        if remove:
            df.drop(col[0], axis=1, inplace=True)
        else:
            print(f"{col[0]} is same as {col[1]}")

def attributeTypes(df):
    df['Class'] = df['Class'].astype('bool')

    # Convert C2 to a boolean
    df.replace({'C2': {'yes': True, 'no': False}}, inplace=True)

    df['C3'] = df['C3'].astype('category')

    df['C4'] = df['C4'].astype('int64')

    df['C5'] = df['C5'].astype('category')
    df['C6'] = df['C6'].astype('category')
    df['C8'] = df['C8'].astype('category')
    df['C12'] = df['C12'].astype('category')
    df['C13'] = df['C13'].astype('category')
    df['C14'] = df['C14'].astype('category')
    
    df['C14'] = df['C18'].astype('category')




# %%
