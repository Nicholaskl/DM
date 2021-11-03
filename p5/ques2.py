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
    'playgolf.python.csv', 
    sep=',',
)

# Convert categorical types to numbers
df["outlook"].replace({"rainy": 0, "overcast": 1, "sunny": 2}, inplace=True)
df["temperature"].replace({"hot": 0, "mild": 1, "cool": 2}, inplace=True)
df["humidity"].replace({"normal": 0, "high": 1}, inplace=True)
df["wind"].replace({"false": 0, "true": 1}, inplace=True)
df["playGolf"].replace({"no": 0, "yes": 1}, inplace=True)

# for classification we need to extract X, y from the dataframe
# for class labels sklearn does not like 'Apple' 'Orange'
# so one needs to encode the labels suitably, for example
y = df.pop('playGolf')
le = LabelEncoder()
y=le.fit_transform(y)
# for data matrix X one needs to convert to numpy
X=df.to_numpy()

# Statify ensures that equal amount of each class is chosen?
# Test size as we want a total of 15 out of 150
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predict_test = model.predict(X_test)
accuracy = accuracy_score(y_test,predict_test)
print("error-rate: {:.1f}%".format((1-accuracy)*100))
print('confusion matrix:\n',confusion_matrix(predict_test, y_test))
print("f score: ", f1_score(y_test, predict_test, average=None))
# %%
