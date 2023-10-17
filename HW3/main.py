import sklearn
import numpy as np
import pandas as pd

# loading in the data
# I added the data cvs to this folder in order to access them
train_data = pd.read_csv("train.csv")
print(train_data.size)


# Checking NaN values for preprocessing #
# print(train_data.isnull().sum())
train_data = train_data.dropna(subset=['Embarked'])

# dropping 'Ticket'
train_data = train_data.drop(columns=['Ticket'])

# Changing the Sex category to numeric
sex_mapping = {'male': 1, 'female': 2}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)


# Changing the Embarked category to numeric
embark_mapping = {'C': 1, 'Q': 2, 'S': 3}
train_data['Embarked'] = train_data['Embarked'].map(embark_mapping)


if __name__ == '__main__':
    print("Colin Houde - HW3 - Machine Learning")


