#import libraries
import pandas as pd
import matplotlib.pyplot as plt

datasets= pd.read_csv('Position_Salaries.csv')
X = datasets.iloc[:,1:2].values
Y = datasets.iloc[:,2].values

#fit the linear regression into the datasets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,Y)

#fit the polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
regressor_2 = LinearRegression()
regressor_2.fit(X_poly, Y)

#visualizing the linear regression results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='orange')


#visualizing the linear regression results
plt.scatter(X, Y, color='purple')
plt.plot(X, regressor_2.predict(X_poly), color='orange')