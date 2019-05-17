import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datasets = pd.read_csv('Position_salaries.csv')
X= datasets.iloc[:, 1:2].values
Y= datasets.iloc[:, 2].values

#fit the random forest regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=0)
regressor.fit(X,Y)


y_pred = regressor.predict(X=6.5)

#plot the random forest  tree regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid= X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y, color='green')
plt.plot(X_grid, regressor.predict(X_grid), color='purple')
plt.title('Truth or Bluff random forest regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()