#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import Datasets
datasets = pd.read_csv('50_Startups.csv')
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

#aAvoiding the dummy variable trap
X = X[:, 1:]

#train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)

#Fit the multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

#building an optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1), dtype=int), values= X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary() 

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()








