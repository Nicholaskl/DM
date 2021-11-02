# AUTHOR: Nicholas Klvana-Hooper
# STUDENT ID: 19872944
# Date Created: 10/9/2021

# Assignment for Data Mining COMP3009 taken during the second semester of 2021.

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

def main():
    """
    The main function of this python file
    This file uses the classic boilerplate style
    You'll find the if __name__ == '__main__': later on in this file to see this
    function be invoked :)
    """

    # Firstly read the csv file and setup empty list
    df = pd.read_csv('data2021.student.csv', sep=',')
    training = []
    test = []

    # Do all preprocessing steps on the dataset
    # This will result in split training and testing sets
    training,test = preProcessing(df)

    # The size of he subset to trst and train on.
    TEST_FRACTION = 0.2
    df = training
    # Make sure class is a category by this point
    df['Class'] = df['Class'].astype('category')

    # Setup smote functions for later
    smote = SMOTE()

    # make sure to have y which is just the class
    # Can't have categorical names so just use label encoder to make these
    # a numerical version of them
    y = df.pop('Class')
    labels = preprocessing.LabelEncoder()
    y = labels.fit_transform(y)
    X = df.to_numpy()
    
    # Use smote here to oversample the data for both sets.
    # This returns the data.
    X_smote, y_smote = smote.fit_resample(X,y)

    # Split data into test and training data from the smote set
    # has a random state I set
    X_train, X_test, y_train, y_test = train_test_split(X_smote, 
                                                        y_smote, 
                                                        test_size=TEST_FRACTION, 
                                                        random_state=2252)

    # Don't need modelTuning when trying to run for final program
    # modelTuning(X_train, X_test, y_train, y_test)
    modelCompare(test, X_train, y_train)

    # Final prediction function
    prediction(test, X_smote, y_smote)


def preProcessing(df):
    """
    This function does all preprocessing steps on the original data
    It will return two sets, the training and testing sets which are used
    for classification later on
    """
    removeIrrelevant(df, remove=True)
    missingEntries(df, remove=True)
    duplicates(df, remove=True)
    attributeTypes(df)
    dataTransformation(df)
    categToNumeric(df)

    return seperateData(df)

def removeIrrelevant(df, remove=False):
    """
    This function removes any irrelevant data from the dataset
    It required a dataframe to work on
    The remove parameter if false, will not actually remove the irrelevant data
    and instead just print out some verbose text. if True, that data will be
    removed instead.
    """
    
    # ID is not needed at all, so remove it
    df.drop(['ID'], axis=1, inplace=True)

    # Gives a series of items with the number of unique values for each
    # attribute, as well as the name of that column
    unique = df.nunique (axis=0)

    # Loops through all of the attributes
    # index is name of column, value is the number of unique values
    for index, value in unique.items():
        # If there is one value, we want to remove it.
        if value == 1:
            if remove:
                df.drop(index, axis=1, inplace=True)
            else:
                # instead of removing, just print out.
                print(f"{index} only has {value} values")

def missingEntries(df, remove=False):
    """
    Will find the missing entries for every attribute
    Prints out attribute name and how many are missings if remove is False
    Otherwise will remove attributes or fill the missing values with the 
    median or mode depending on the data type
    """

    # Loop through all the attributes
    attribute_names = df.columns
    for attr in attribute_names:
        # Find the sum of all null values - missing values
        sum = df[attr].isnull().sum()

        # If there are missing values then do more checks
        if sum > 1:
            # If remove is true the do code that would remove data from the
            # dataframe or otherwise just print out missing values.
            if remove:
                # Remove entries with more than 80% missing values
                # But don't remove the class column!!!
                if (sum/df.shape[1] > 0.8) and (attr != 'Class'):
                    # Remove them from the dataframe
                    df.drop(attr, axis=1, inplace=True)
                elif (sum/df.shape[1] > 0) and (attr != 'Class'):
                    # Get the datatype of the column
                    dt = df[attr].dtype

                    # If it is numeric then replace missing values with the mean
                    if(dt == np.int64 or dt == np.float64):
                        df[attr].fillna(df[attr].mean(), inplace=True)
                    # If it is categorical then replace misssing with the mode
                    elif (df[attr].dtype == object):
                        df[attr].fillna(df[attr].mode()[0], inplace=True)
            else:
                print(attr + " " + str(sum))
    


def duplicates(df, remove=False):
    """
    Find all of the duplicate rows and columns in the dataframe
    If remove is True, duplicates will be dropped from the data
    Otherwise some printing occurs to show missing data
    """

    # Get all the duplicates in the data
    duplicates_in_df = df.duplicated(keep='first', subset=None)
    # Get the specific duplicate rows from the data
    duplicate_rows = np.where(duplicates_in_df == True)[0]

    # Go through all duplicate rows
    for row in duplicate_rows:
        # if remove is true then drop from dataframe otherwise just print
        if remove:
            df.drop(index=row, axis=0, inplace=True)
        else:
            print("Duplicate row", row)
        
    # Find duplicate attributes
    duplicate_cols = []
    # loop through columns in the dataframe
    for col in range(df.shape[1]): 
        # Select only the current column
        curr_col = df.iloc[:,col]

        # Now loop through each subsequent column that comes after
        for checkCol in range(col+1, df.shape[1]):
            # Get the next specific column to check against
            check_col = df.iloc[:, checkCol]

            # Check if the columns match
            # Have to do this strange equality on arrays
            if (curr_col.values == check_col.values).all():
                # Add the duplicate column to the list
                duplicate_cols.append([curr_col.name, check_col.name])

    # Go through all of the duplicate columns.
    # if remove is true then remove, otherwise just print
    for col in duplicate_cols:
        if remove:
            df.drop(col[0], axis=1, inplace=True)
        else:
            print(f"{col[0]} is same as {col[1]}")

def attributeTypes(df):
    """
    Will set the correct attribute types for the data inside of the dataframe
    """

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
    """
    Transforms the numeric values in the dataset. In this case I use a min-max
    scaler to scale the data more appropriatly
    """

    # Get the columns that are numerical
    # In this case it gets C1, C4, C16, C19, C25 and C31

    normal = ['C19']
    skewed = ['C1', 'C4', 'C16', 'C25', 'C31']

    # Use the Standard scaler to only scale the mostly normal data attribute
    scale = preprocessing.StandardScaler()
    scaled_data = pd.DataFrame(scale.fit_transform(df.loc[:,normal].values), columns = normal)
    df.loc[:,normal] = scaled_data.values

    # Have to scale the skewed data differently 
    # Use the powerTransformer with box-cox to convert the log-normal distribution
    # to a normal distribution here.
    scale = preprocessing.PowerTransformer(method='box-cox')
    scaled_data = pd.DataFrame(scale.fit_transform(df.loc[:,skewed].values), columns = skewed)
    df.loc[:,skewed] = scaled_data.values

    numeric = ['C1', 'C4', 'C16', 'C19', 'C25', 'C31']

    # uses a min max scaler for all of the attributes now
    scale = preprocessing.MinMaxScaler()
    scaled_data = pd.DataFrame(scale.fit_transform(df.loc[:,numeric].values), columns = numeric)
    df.loc[:,numeric] = scaled_data.values

def seperateData(df):
    """
    Seperated the main dataframe into the two sets for training and testing
    testing in this case is the unlabelled data.
    """
    training = df.iloc[:-100]
    test = df.iloc[-100:]

    return training,test

def categToNumeric(df):
    """
    Becuase of kNN this is needed as all datatypes now need to be numerical
    So this converts all caterocial data into numeric values
    And also booleans into 1s and 0s
    """
    numeric = df.select_dtypes(include='category')
    numeric.replace({'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9, 'V10': 10}, inplace=True)
    # numeric = numeric.astype('int64')
    df[numeric.columns] = numeric

    # Convert all booleans into 0s and 1s
    bools = df.select_dtypes(include='bool')
    bools = bools.astype(int)

    df[bools.columns] = bools

def modelTuning(X_train, X_test, y_train, y_test):
    """
    Function used to find the best hyperparameters for each of the models used
    knn, naive bayes and decision tree
    """
    kmin = 7
    kmax = 100

    # sets up the parameters needed for the knn classifier.
    params = {'n_neighbors' : list(range(kmin,kmax, 2)),
                'p': [1, 2],
                'weights': ['uniform', 'distance']}
    # runs a function to test the parameters and find the best one
    modelParams(X_train, X_test, y_train, y_test, KNeighborsClassifier(), params)

    # now do the same for the decision tree
    params = {'min_samples_split': list(range(2,15))}
    modelParams(X_train, X_test, y_train, y_test, DecisionTreeClassifier(random_state=246), params)

    # and finally same for the naive bayes model
    params = {'var_smoothing': np.logspace(0,-8, num=40)}
    modelParams(X_train, X_test, y_train, y_test, GaussianNB(), params)

def modelParams(X_train, X_test, y_train, y_test, model, params):
    """
    This function handles that actual checks of paramters on each of the 
    models that are used. It will output the accuracy and best paramters
    that can be used as well as other handy information.
    """

    # Setup the grid search with the parameters for the specified model
    gridSearch = GridSearchCV(model, param_grid=params)

    # serform the grid search on the training set
    gridSearch.fit(X_train, y_train)

    # Outputs the parameters and best scoring model set
    bestSearch = gridSearch.best_estimator_
    print('Model score:', gridSearch.best_score_)
    print('Model parameters:', gridSearch.best_params_)
    print('Model classifier:', bestSearch)

    # Now use this to predict the test set
    predict_test = gridSearch.predict(X_test)
    # And socre it on the y labels already decided
    accuracy = accuracy_score(y_test,predict_test)

    # Need to convert to list to run counts
    # Output how many of each 1 and 1 there is
    # Also show the confusion matrix
    predic = predict_test.tolist()
    print("Accuracy on test set:", accuracy)
    print("0s:", predic.count(0), "1s:", predic.count(1))
    print('Confusion matrix:',confusion_matrix(predict_test, y_test))

def modelCompare(test, X_train, y_train):
    """
    This is the main function that deals with comparing all 3 of the models
    with each other. It outputs the metrics used to compare them to each other
    """

    # Firstly do some setup on the final set making sure to seperate 
    # the class labels
    final_y = test.pop('Class')
    final_X = test.to_numpy()

    # here is a list of the models with their best parameters discovered
    models = [KNeighborsClassifier(n_neighbors=7, weights='uniform', p=1),
        DecisionTreeClassifier(min_samples_split=5, random_state=246),
        GaussianNB(var_smoothing=0.03665241237079628)]

    # sets up empty data frames for the output of the tests
    scores = pd.DataFrame()
    numZeroOne = pd.DataFrame()

    # go through each model
    for model in range(len(models)):
        # Run the scording of accuracy on each one using cross validation
        # This performs on the training data over a specified 10 folds.
        scores[model, "acc"] = cross_val_score(models[model], X_train, y_train, scoring ='accuracy', cv = 10)

        # Then fits again on the whole data
        # this is in order to find a prediction on the final data
        models[model].fit(X_train, y_train)
        test_prediction = models[model].predict(final_X)

        # Gets the number of zeros and ones from the predicted 
        zero = np.bincount(test_prediction.astype(int))[0]
        one = np.bincount(test_prediction.astype(int))[1]

        # Adds these to the dataframe output
        numZeroOne[model] = [zero, one]
    
    # Outputs all of the data
    print("===================")
    print(scores)
    print(numZeroOne)
    # numZeroOne.columns = ["knn", "decision tree", "naive bayes"]
    # scores.groupby('Class').size().plot.bar()
    # plt.bar(["0.0", "1.0"], [650, 650])
    # numZeroOne.plot.bar()
    # plt.ylabel('Number of times present')
    # plt.xlabel('Models')
    # plt.title('Number of 0s and 1s in model predictions')
    # plt.show()

def prediction(X_test, x, y):
    """
    This handles the final prediction of the unlabelled data
    """

    # Double check everything is fine at this point
    X_test = X_test.to_numpy()

    # First run the knn model
    kNN = KNeighborsClassifier(n_neighbors=7, weights='uniform', p=1)
    # Should be fit on the original sample not subsets
    kNN.fit(x, y) 
    y_kNN = kNN.predict(X_test) # predict on the unlabeled data
    y_kNN= y_kNN.tolist()

    # Now run on the decision tree model
    dT = DecisionTreeClassifier(min_samples_split=5, random_state=246)
    # Should be fit on the original sample not subsets
    dT.fit(x, y) 
    y_dT = dT.predict(X_test) # predict on the unlabeled data
    y_dT= y_dT.tolist()

    # Open the file and ensure its utf-8
    output = open("predict.csv", "w", encoding="utf-8")
    # Needs this as the top of the file.
    output.write("ID,Predict1,Predict2" + "\n")

    # Outputs all of the predictions for both models
    for num in range(100):
        output.write(str(num+1001) + ',' + str(y_kNN[num])+',' + str(y_dT[num]) + '\n')

    # Outputs the balance of the two predictions
    print('KNN == 0s:', y_kNN.count(0),'1s:', y_kNN.count(1))
    print('DT == 0s:', y_dT.count(0),'1s:', y_dT.count(1))

    # Stop writing to the file
    output.close()

# This is standard boilerplate python to check for main to avoid
# certain side effects.
if __name__ == '__main__':
    main()
