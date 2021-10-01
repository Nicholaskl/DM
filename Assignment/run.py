#%%
import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import arff

def main():

    df=pd.read_csv('data2021.student.csv', sep=',')
    training = []
    test = []

    rows = df.shape[0]
    training,test = preProcessing(df)

    arff.dump('training_two.arff',training.values,relation="train", names=df.columns)
    arff.dump('testing_two.arff',test.values,relation="test", names=df.columns)

    return df

def preProcessing(df):
    removeIrrelevant(df, remove=True)
    missingEntries(df, remove=True)
    duplicates(df, remove=True)
    attributeTypes(df)
    dataTransformation(df)
    categToNumeric(df)
    training,test = seperateData(df)

    return training, test


    
def categToNumeric(df):
    numeric = df.select_dtypes(include='category')
    numeric.replace({'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9, 'V10': 10}, inplace=True)
    # numeric = numeric.astype('int64')
    df[numeric.columns] = numeric

    bools = df.select_dtypes(include='bool')
    bools = bools.astype(int)

    df[bools.columns] = bools

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
                        df[attr].fillna(df[attr].mode()[0], inplace=True)
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
    df['Class'] = df['Class'].astype('category')

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

    df['C18'] = df['C18'].astype('category')

    df.replace({'C20': {1: 'V1', 2: 'V2', 3: 'V3', 4: 'V4'}}, inplace=True)
    df['C20'] = df['C20'].astype('category')
    
    df.replace({'C22': {'V1': True, 'V2': False}}, inplace=True)

    df.replace({'C23': {1: 'V1', 2: 'V2', 3: 'V3', 4: 'V4'}}, inplace=True)
    df['C23'] = df['C23'].astype('category')

    df['C24'] = df['C24'].astype('category')

    df['C26'] = df['C26'].astype('category')

    df.replace({'C27': {1: 'V1', 2: 'V2', 3: 'V3', 4: 'V4'}}, inplace=True)
    df['C27'] = df['C27'].astype('category')

    df['C28'] = df['C28'].astype('category')

    # Have to round here as the mean was used so there are some
    # average values whacked in here somewhere
    df['C29'] = df['C29'].round()
    df.replace({'C29': {1.0: True, 2.0: False}}, inplace=True)
    df['C29'].mode()

def dataTransformation(df):
    # Get the columns that are numerical
    numeric = df.select_dtypes(include='number')

    scaled = MinMaxScaler().fit_transform(numeric)

    df[numeric.columns] = scaled

def seperateData(df):
    training = df.iloc[:-100]
    test = df.iloc[-100:]

    return training,test

df = main()


# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.io import arff
from imblearn.over_sampling import SMOTE
from collections import Counter

kmin = 1
kmax = 18
test_fraction = 0.15

# load data
data_file='training_two.arff'
data, meta = arff.loadarff(data_file)
df=pd.DataFrame(data=data)
df['Class']=df['Class'].astype('category')

class_count_0, class_count_1 = df['Class'].value_counts()
class_0 = df[df['Class'] == 0]
class_1 = df[df['Class'] == 1]
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)

smote = SMOTE()


# for classification we need to extract X, y from the dataframe
# for class labels sklearn does not like 'Apple' 'Orange'
# so one needs to encode the labels suitably, for example
y = df.pop('Class')
le = LabelEncoder()
y=le.fit_transform(y)

# for data matrix X one needs to convert to numpy
x=df.to_numpy()

x_smote, y_smote = smote.fit_resample(x,y)
print('Origianl dataset shape:', Counter(y))
print('Resampple dataset shape:', Counter(y_smote))

# to split the data into different sets one may use train_test_split
# stratify ensures we have consistent proportion of classes
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=test_fraction, stratify=y_smote)

model = KNeighborsClassifier()
skf = StratifiedKFold(n_splits=9)

# set up the search
k_range = list(range(kmin,kmax))
params = {'n_neighbors' : k_range}
gridSearch = GridSearchCV(model, param_grid=params, cv=skf)

# serform the grid search
gridSearch.fit(x_train, y_train)

# obtain the best candidate
best_kNN = gridSearch.best_estimator_

# Display the best model scores and parameters
print('Best score: {}'.format(gridSearch.best_score_))
print('Best parameters: {}'.format(gridSearch.best_params_))
print('Best classifier: {}'.format(best_kNN))


KNeighborsClassifier(n_neighbors=gridSearch.best_params_['n_neighbors'])
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

# load data
data_file='testing_two.arff'
data, meta = arff.loadarff(data_file)
df_test=pd.DataFrame(data=data)

# for data matrix X one needs to convert to numpy
y = df_test.pop('Class')
x=df_test.to_numpy()

y = model.predict(x)

output = open("robot.csv", "w")
y= y.tolist()

for num in range(100):
    output.write(str(num+1001) + ',' + str(y[num])+'\n')

print('0s:', y.count(0))
print('1s:', y.count(1))

output.close()
# %%
