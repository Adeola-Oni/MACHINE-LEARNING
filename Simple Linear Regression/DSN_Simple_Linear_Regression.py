# import libraries
import pandas as pd
import matplotlib.pyplot as plt

#get datasets
datasets = pd.read_csv('Salary_Data.csv');
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:,1].values

#split dataset into test and train sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=0)

#linear regression part
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#prediction the 
y_pred = regressor.predict(X_test)

#visualizing the test result
plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('sal')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()