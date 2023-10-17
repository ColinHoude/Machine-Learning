import sklearn
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

# loading in the data
# I added the data cvs to this folder in order to access them
train_data = pd.read_csv("train.csv")


# Checking NaN values for preprocessing #
# print(train_data.isnull().sum())
train_data = train_data.dropna(subset=['Embarked'])

# dropping 'Ticket'
train_data = train_data.drop(columns=['Ticket'])

# dropping 'Name'
train_data = train_data.drop(columns=['Name'])


# Changing the Sex category to numeric
sex_mapping = {'male': 1, 'female': 2}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)

# Changing the Embarked category to numeric
embark_mapping = {'C': 1, 'Q': 2, 'S': 3}
train_data['Embarked'] = train_data['Embarked'].map(embark_mapping)


# Change the 'Cabin' to just A,B,C,D... instead of A23, B16, C87
train_data["Cabin"] = train_data["Cabin"].astype(str).str[:1]

# Change the Cabin category to numeric,
# Setting them to decreasing order as cabin A should have a higher influence than cabin G
cabin_mapping = {'n': 0, 'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
train_data['Cabin'] = train_data['Cabin'].map(cabin_mapping)


# Below is the feature selection section #
# tempTrainData = train_data.dropna()
# featSelectX = tempTrainData.drop(columns=['Survived', 'PassengerId'])
# featSelectY = tempTrainData['Survived']


# Select top x features based on mutual info regression
# X_clf_new = SelectKBest(score_func=chi2, k=1).fit_transform(featSelectX, featSelectY)
# print(X_clf_new[:10])
# print(featSelectX.head)

trainDataFeatures = ['Fare', 'Cabin', 'Sex', 'Age', 'Pclass']

finalTrainDataX = train_data[trainDataFeatures]
finalTrainDataY = train_data['Survived']


clf = DecisionTreeClassifier()

# train the classifier
clf = clf.fit(finalTrainDataX, finalTrainDataY)

y_pred = clf.predict(finalTrainDataX)

print(("Accuracy:", metrics.accuracy_score(finalTrainDataY, y_pred)))

if __name__ == '__main__':
    print("Colin Houde - HW3 - Machine Learning")


