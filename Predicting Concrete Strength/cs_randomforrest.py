# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# read the data
df = pd.read_csv('Concrete_Data_Yeh.csv')

# split the data into features and target
X = df.drop('csMPa', axis=1)
y = df['csMPa']

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to search
param_grid = {'n_estimators': [200, 500], 'max_depth': [8, 10], 'min_samples_split': [2, 4, 6], 'min_samples_leaf': [1, 2, 4]}

# create a random forest regressor
model = RandomForestRegressor(random_state=42)

# set the parameters on grid search
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)

# run the grid search
grid_search.fit(X_train, y_train)

# print the best parameters
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)


# fit the random forest model on the training data
model.fit(X_train, y_train)


# get the predictions
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# evaluate the model
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))


# Calculate the 95% confidence interval for predictions
n = len(y_pred)
m = np.mean(y_pred)
std_err = np.std(y_pred)
h = std_err * 1.96

start = m - h
end = m + h

print('Start:', start)
print('End:', end)











