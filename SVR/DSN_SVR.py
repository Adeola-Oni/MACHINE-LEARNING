#import the libraries
import pandas as pd
import matplotlib.pyplot as plt

#get the datasets
datasets= pd.read_csv('Position_Salaries.csv')
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:, 2].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X) 
Y = sc_Y.fit_transform(X) 


#fit the SVR to the dataset
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X,Y)

#predict a new result
y_pred = svr_regressor.predict(X=6.5)

#plot the graph
plt.scatter(X,Y, color='green')
plt.plot(X, svr_regressor.predict(X), color='purple')
plt.title('Truth or Bluff SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()