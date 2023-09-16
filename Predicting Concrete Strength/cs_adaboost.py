import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# Read the data

df = pd.read_csv('Concrete_Data_Yeh.csv')

# Split the data into training and testing sets

X = df.drop('csMPa', axis=1)
y = df['csMPa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to random search

param_dist = {'n_estimators': [200, 400, 600],
'learning_rate': [0.01, 0.1, 1],
'loss': ['linear', 'square', 'exponential']}

# Create the model

model = AdaBoostRegressor(random_state=42)

# Set the parameters on random search

random_search = RandomizedSearchCV(model,
param_distributions=param_dist,
n_iter=10,
scoring='neg_mean_squared_error',
cv=5,
verbose=2,
n_jobs=-1,
random_state=42)

# Run the random search

random_search.fit(X_train, y_train)

# Print the best parameters

print(random_search.best_params_)
print(random_search.best_score_)
print(random_search.best_estimator_)

# fit the model on the training data with best hyperparameters

model = AdaBoostRegressor(**random_search.best_params_, random_state=42)
model.fit(X_train, y_train)


# Predict the test set results

y_pred = model.predict(X_test)

# Evaluate the model performance

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Calculate the 95% confidence interval for predictions
n = len(y_pred)
s = np.std(y_test - y_pred, ddof=1)
t_value = 1.96  # for a 95% confidence interval and n > 30
confidence_interval = (y_pred - t_value * s / np.sqrt(n), y_pred + t_value * s / np.sqrt(n))
print("95% confidence interval:", confidence_interval)
