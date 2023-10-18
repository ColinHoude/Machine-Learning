import sklearn
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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

# drop the 1 cabin row that is Nan
train_data = train_data.dropna(subset=['Cabin'])

# get the mean of the ages
mean_age = train_data['Age'].mean()

# fill all the NaN age values with the mean
train_data['Age'].fillna(value=mean_age, inplace=True)

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

X_train, X_test, Y_train, Y_test = train_test_split(finalTrainDataX, finalTrainDataY, test_size=0.2, random_state=42)

# Below is fine-tuning the model and picking the best hyperparameters
clf = DecisionTreeClassifier()
clf2 = RandomForestClassifier()

# Bagging classifier
bagging_classifier = BaggingClassifier(clf, n_estimators=10, random_state=42)

# Adaboost classifier
ada_class = AdaBoostClassifier(clf, n_estimators=10, random_state=42)


# # train the classifier
clf = clf.fit(X_train, Y_train)
clf2 = clf2.fit(X_train, Y_train)
bagging_classifier = bagging_classifier.fit(X_train, Y_train)
ada_class = ada_class.fit(X_train, Y_train)


# Plot the decision tree -- I commented this out because it was unnecessary when doing cross validation
# plot_tree(clf, filled=True, feature_names=trainDataFeatures, class_names=["0", "1"])
# plt.show()


# below is the five-fold cross validation
scores = cross_val_score(clf, finalTrainDataX, finalTrainDataY, cv=5)
scores2 = cross_val_score(clf2, finalTrainDataX, finalTrainDataY, cv=5)
bagScore = bagging_classifier.score(X_test, Y_test)
adaScore = ada_class.score(X_test, Y_test)


print("Decision Tree Classifier: %0.5f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("Random Forest Classifier: %0.5f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))
print("Bagging Accuracy:", bagScore)
print("AdaBoost Accuracy:", adaScore)

if __name__ == '__main__':
    print()
    print("Colin Houde - HW3 - Machine Learning")


