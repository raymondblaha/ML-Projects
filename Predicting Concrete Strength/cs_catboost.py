# installed catboost. Note: catboost is not available on python 3.11 MacoOS. Running on python 3.10.
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


# Read in data
df = pd.read_csv('Concrete_Data_Yeh.csv')

# Now it is time to split the data into training and testing sets
X = df.drop('csMPa', axis=1)
y = df['csMPa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters to be tuned
params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [3, 5, 7],
    'iterations': [100, 500, 1000]
}

# Perform grid search to find the best hyperparameters
model = CatBoostRegressor()
grid_search_result = model.grid_search(params, X_train, y_train, cv=5)

# Fit the model with the best hyperparameters
best_model = CatBoostRegressor(learning_rate=grid_search_result['params']['learning_rate'],
                                depth=grid_search_result['params']['depth'],
                                iterations=grid_search_result['params']['iterations'])
best_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the best hyperparameters and expected RMSE error
print("Best hyperparameters:", grid_search_result['params'])
print("Expected RMSE error:", rmse)
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))


# Calculate the 95% confidence interval for predictions
n = len(y_pred)
s = np.std(y_test - y_pred, ddof=1)
t_value = 1.96  # for a 95% confidence interval and n > 30
confidence_interval = (y_pred - t_value * s / np.sqrt(n), y_pred + t_value * s / np.sqrt(n))
print("95% confidence interval:", confidence_interval)