#%%

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# for model selection with cross-validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

data_file='iris.2D.arff'
data, meta = arff.loadarff(data_file)
df=pd.DataFrame(data=data)
df['class']=df['class'].astype('category')

# for classification we need to extract X, y from the dataframe
# for class labels sklearn does not like 'Apple' 'Orange'
# so one needs to encode the labels suitably, for example
y = df.pop('class')
le = LabelEncoder()
y=le.fit_transform(y)
# for data matrix X one needs to convert to numpy
X=df.to_numpy()

# Statify ensures that equal amount of each class is chosen?
# Test size as we want a total of 15 out of 150
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predict_test = model.predict(X_test)
accuracy = accuracy_score(y_test,predict_test)
print("error-rate: {:.1f}%".format((1-accuracy)*100))
print('confusion matrix:\n',confusion_matrix(predict_test, y_test))
print("f score: ", f1_score(y_test, predict_test, average=None))


def findBestParameters(df):

    # kNN with 9-fold cross validation - the AUTO approach
    # you can use GridSearchCV to search for best models
    # and with cross-validation    
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    kmin = 1
    kmax = 9

    model = KNeighborsClassifier()
    skf = StratifiedKFold(n_splits=9)
    from sklearn.model_selection import StratifiedKFold, KFold
    # set up the search
    k_range = list(range(kmin, kmax, 2))
    params = {'n_neighbors' : k_range}
    gridSearch = GridSearchCV(model, param_grid=params, cv=skf)

    # serform the grid search
    gridSearch.fit(X_train, y_train)

    # obtain the best candidate
    best_kNN = gridSearch.best_estimator_

    # Display the best model scores and parameters
    print('Best score: {}'.format(gridSearch.best_score_))
    print('Best parameters: {}'.format(gridSearch.best_params_))
    print('Best classifier: {}'.format(best_kNN))

# %%
