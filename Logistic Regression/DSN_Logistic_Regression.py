# Logistic Regression

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4]
Y = dataset.iloc[:, 4]

# train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
standardscalar_X = StandardScaler()
X_train = standardscalar_X.fit_transform(X_train)
X_test = standardscalar_X.transform(X_test)

# fitting the regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

#predict a result
y_pred = classifier.predict(X_test)

#make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)