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
    'crx.csv', 
    sep=',',
    na_values='?'
)

# Chance data types
df["A1"].replace({"b": 0, "a": 1}, inplace=True)
df["A4"].replace({"u": 0, "y": 1, "l":2, "t":3}, inplace=True)
df["A5"].replace({"g": 0, "p": 1, "gg":2}, inplace=True)
df["A6"].replace({"c": 0, "d": 1, "cc":2, "i":3, "j":4, "k":5,"m":6,"r":7,"q":8,"w":9,"x":10,"e":11,"aa":12,"ff":13}, inplace=True)
df["A7"].replace({"v": 0, "h": 1, "bb":2, "j":3,"n":4,"z":5,"dd":6,"ff":7,"o":8}, inplace=True)
df["A9"].replace({"t":0, "f":1}, inplace=True)
df["A10"].replace({"t":0, "f":1}, inplace=True)
df["A12"].replace({"t":0, "f":1}, inplace=True)
df["A13"].replace({"g":0, "p":1,"s":2}, inplace=True)
df["A16"].replace({"-":0, "+":1}, inplace=True)

# axis=0 for rows
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# for classification we need to extract X, y from the dataframe
# for class labels sklearn does not like 'Apple' 'Orange'
# so one needs to encode the labels suitably, for example
y = df.pop('A16')
le = LabelEncoder()
y=le.fit_transform(y)
# for data matrix X one needs to convert to numpy
X=df.to_numpy()

# Statify ensures that equal amount of each class is chosen?
# Test size as we want a total of 15 out of 150
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predict_test = model.predict(X_test)
accuracy = accuracy_score(y_test,predict_test)
print("error-rate: {:.1f}%".format((1-accuracy)*100))
print('confusion matrix:\n',confusion_matrix(predict_test, y_test))
print("f score: ", f1_score(y_test, predict_test, average=None))
# %%
