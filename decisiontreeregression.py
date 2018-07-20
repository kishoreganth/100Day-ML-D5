
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# fitting the decision tree to the dataset 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# predicting the result using the regressor 
y_pred = regressor.predict(6.5) 

# visualising the result 
plt.scatter(X, y,color = 'red')
plt.plot(X,regressor.predict(X),color ='blue')
plt.title('Truth or Bluff DecisionTree Regression')
plt.xlabel('Position Salary')
plt.ylabel('Salary')
plt.show()


#visualising the result with high resolution because Decision Tree Regressor is non continous
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X, y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color ='blue')
plt.title('DecisionTree Regression')
plt.xlabel('Position Salary')
plt.ylabel('Salary')
plt.show()
