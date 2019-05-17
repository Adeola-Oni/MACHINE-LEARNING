# import the libraries #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the datasets #
datasets = pd.read_csv('Data.csv')
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 3].values

#substitute the missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Change the format of the Ctergorical data to numreric data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#SPlitting into test_train set
from sklearn.cross_validation import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit(X_test)