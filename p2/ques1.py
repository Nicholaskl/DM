#%%

from numpy.core.fromnumeric import shape
from numpy.core.records import find_duplicate
import pandas as pd
import numpy as np
from pandas.core.arrays.categorical import contains
import arff

## Read the csv file. Can do `index_col=8` to pick important column
# Last column is the dependent column
df = pd.read_csv(
    'D:\CURTIN\YEAR3\Sem2\DM\p2\\arrhythmia.data.csv', 
    sep=',',
    header=None, # There are no column names... 
    na_values='?' # Empty values are shown as a '?' in the csv
)

## To only show the top 10 rows
pd.set_option('display.max_columns', 500)


## Show number of attributes and instances
print("Numer of instances: ", df.shape[0])
print("Numer of attributes: ", df.shape[1])

# Created a list of all the names 1 to number of attributes we have
names = ['attr' + str(num + 1) for num in range(0,df.shape[1])]

# Now set column names to the ones generated
df.columns = names

# Print top 10 instances
df.head(10)

#%%

# Modifying the class attribute mapping

# df.describe(include=[np.number]) This proved 'attr280' is numerical

df['attr280']=df['attr280'].astype('category')

# Dictionary for values 2-15 that should map to 'A'
tmp = {}
keys = range(2,16) # Going from 2 to 16-1 for the keys
for key in keys:
    tmp[key] = 'A'

# These are the other two values that will be changed
tmp[1] = 'N'
tmp[16] = 'O'

change = {'attr280': tmp} #Only column 'attr280'

# Change the 280 column to the new set of
df.replace(change, inplace=True)

df['attr280'].head(10)


# %%

# Convert set of columns to nominal
# att2, att23, att25, att27, att35, att37, att39

df['attr2']=df['attr2'].astype('category')
df['attr23']=df['attr23'].astype('category')
df['attr25']=df['attr25'].astype('category')
df['attr27']=df['attr27'].astype('category')
df['attr35']=df['attr35'].astype('category')
df['attr37']=df['attr37'].astype('category')
df['attr29']=df['attr29'].astype('category')

df['attr2'].head(10)

# %%

# Find and remove attributes with more than 80% missing values

attribute_names = df.columns
for attr in attribute_names:
    num_missing = df[attr].isnull().sum() # Gets sum of missing elements in column

    if (num_missing/df.shape[0]) > 0.8: # If 80% are missing, remove
        print('Dropped Column:', attr)
        df.drop(columns=attr, inplace=True)
    # Find all attributes with less than 5% missing values and replace these missing values with either
    # the mean and the mode of the attribute.
    elif (num_missing/df.shape[0] < 0.05) and num_missing != 0:
        print(f'Altered: {attr}')
        dt = df[attr].dtype
        if(dt == np.int64 or dt == np.float64):
            # in place means don't have to set the variable
            df[attr].fillna(df[attr].mean(), inplace=True)
        elif (df[attr].dtype == object):
            df[attr].fillna(df[attr].mode(), inplace=True)

df['attr15'].head(15)

# %%

# Discretize attributes att3 and att4 into 10 equiwidth ranges and 10 equi-depth ranges respectively.
# Examine and comment on the intervals in each case
df['attr3'].head(10)

new_labels = ['D' + str(i) for i in range(10)]
df['attr3']=pd.qcut(x=df['attr3'],
                        q = 11,
                        labels=new_labels,
                        duplicates='drop')

new_labels = ['D' + str(i) for i in range(10)]
print(new_labels)
df['attr4']=pd.qcut(x=df['attr4'],
                        q = 10,
                        labels=new_labels,
                        duplicates='drop')

# %%

#  Standardise all numeric attributes to a mean of 0 and a standard deviation of 1

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# StandardScaler ensure data has mean 0 and s.d. of 1
df_numeric = df.select_dtypes(include='number')
scaled_df_numeric = StandardScaler().fit_transform(df_numeric)
df[df_numeric.columns] = scaled_df_numeric

# scaled_df_numeric = MinMaxScaler().fit_transform(df_numeric)
# df[df_numeric.columns] = scaled_df_numeric

# %%

# Detect all duplicate rows and remove them if found. None in this file...

duplicates = df.duplicated(subset=None, keep='first')
duplicate_rows = np.where(duplicates == True)#[0]

for duplicate_row in duplicate_rows:
    print("Dropped row", duplicate_row)
    df.drop(index=duplicate_row, axis=0, inplace=True)



# %%

# Randomply sample 100 instances and save them as test.arff. Save the remaining instances
# as train.arff.

# `sample` will sample a fraction of the datafram... in this case '1' so 100%
# The reset index is to ensure there are no indexes carried over
# Without reset original indexes can be seen in the sampled data

df.sample(frac=1).reset_index(drop=True)

test = df.iloc[:100] # This just grabs the first 100 elements
train = df.iloc[100:]

# Makes an arff file? 
arff.dump('test.arff', test.values, relation='test', names = df.columns)
arff.dump('train.arff', train.values, relation='train', names = df.columns)

# %%

