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

df = pd.read_csv(
    'bank.csv', 
    sep=',',
    na_values='?'
)

df.drop(['id'], axis=1, inplace=True)
df["sex"].replace({"FEMALE":0, "MALE":1}, inplace=True)
df["region"].replace({"INNER_CITY":0, "TOWN":1, "RURAL":2, "SUBURBAN":3}, inplace=True)

df["married"].replace({"NO":0, "YES":1}, inplace=True)
df["car"].replace({"NO":0, "YES":1}, inplace=True)
df["save_act"].replace({"NO":0, "YES":1}, inplace=True)
df["current_act"].replace({"NO":0, "YES":1}, inplace=True)
df["mortgage"].replace({"NO":0, "YES":1}, inplace=True)
df["pep"].replace({"NO":0, "YES":1}, inplace=True)

# Put age into 4 equi-width bins - qcut for same amount of instances
df['age'] = pd.cut(df['age'], bins=4, labels=[1,2,3,4])

test_set = df.iloc[:-100, :]
final_set = df.iloc[-100:,:]

# for classification we need to extract X, y from the dataframe
# for class labels sklearn does not like 'Apple' 'Orange'
# so one needs to encode the labels suitably, for example
y = test_set.pop('pep')
le = LabelEncoder()
y=le.fit_transform(y)
# for data matrix X one needs to convert to numpy
X=test_set.to_numpy()

final_y = final_set.pop('pep')
le = LabelEncoder()
final_y=le.fit_transform(final_y)
# for data matrix X one needs to convert to numpy
final_X=final_set.to_numpy()

print("\n=== Naive Bayes ===")
model = GaussianNB()
model.fit(X, y)
predict_test = model.predict(final_X)
accuracy = accuracy_score(final_y,predict_test)
print("error-rate: {:.1f}%".format((1-accuracy)*100))
print('confusion matrix:\n',confusion_matrix(predict_test, final_y))
print("f score: ", f1_score(final_y, predict_test, average=None))

print("\n=== Decision Tree ===")
model = DecisionTreeClassifier()
model.fit(X, y)
predict_test = model.predict(final_X)
accuracy = accuracy_score(final_y,predict_test)
print("error-rate: {:.1f}%".format((1-accuracy)*100))
print('confusion matrix:\n',confusion_matrix(predict_test, final_y))
print("f score: ", f1_score(final_y, predict_test, average=None))

print("\n=== KNeighboursClassifier ===")
model = KNeighborsClassifier(n_neighbors=69)
model.fit(X, y)
predict_test = model.predict(final_X)
accuracy = accuracy_score(final_y,predict_test)
print("error-rate: {:.1f}%".format((1-accuracy)*100))
print('confusion matrix:\n',confusion_matrix(predict_test, final_y))
print("f score: ", f1_score(final_y, predict_test, average=None))


# kNN with 9-fold cross validation - the AUTO approach
# you can use GridSearchCV to search for best models
# and with cross-validation    
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

kmin = 1
kmax = 100

model = KNeighborsClassifier()
skf = StratifiedKFold(n_splits=9)
from sklearn.model_selection import StratifiedKFold, KFold
# set up the search
k_range = list(range(kmin, kmax, 2))
params = {'n_neighbors' : k_range}
gridSearch = GridSearchCV(model, param_grid=params, cv=skf)

# serform the grid search
gridSearch.fit(X, y)

# obtain the best candidate
best_kNN = gridSearch.best_estimator_

# Display the best model scores and parameters
print("\n=== Best KNN ===")
print('Best score: {}'.format(gridSearch.best_score_))
print('Best parameters: {}'.format(gridSearch.best_params_))
print('Best classifier: {}'.format(best_kNN))
# %%
