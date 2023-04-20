import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load the Boston Housing dataset
df = pd.read_csv('BostonHousing.csv')

# drop any rows with missing values
df.dropna(inplace=True)

# split the data into features (X) and target (y)
X = df.drop('medv', axis=1)
y = df['medv']

# split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create a linear regression object
lr = LinearRegression()

# train the model on the training data
lr.fit(X_train, y_train)

# make predictions on the test data
y_pred = lr.predict(X_test)

# calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# calculate the R-squared score
r2 = r2_score(y_test, y_pred)

print('Mean squared error: ', mse)
print('R-squared score: ', r2)
