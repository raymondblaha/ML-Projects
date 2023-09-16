import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

# read the data
df = pd.read_csv('Concrete_Data_Yeh.csv')

# split the data into features and target
X = df.drop('csMPa', axis=1)
y = df['csMPa']

# process the data so that the ytrain and ytest are binary
y = np.where(y > 35, 1, 0)

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create the model
model = SVC(kernel='rbf', random_state=42)

# set the parameters on grid search
param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.1, 1, 10, 100], 'degree': [3, 4, 5], 'coef0': [0, 1, 2]}
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

# run the grid search
grid_search.fit(X_train, y_train)

# print the best parameters
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)


# fit the model on the training data with the best parameters
model = grid_search.best_estimator_
model.fit(X_train, y_train)

# predict on the test data
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Expected RMSE error:", rmse)
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))


# Calculate the 95% confidence interval for predictions
n = len(y_pred)
m = np.mean(y_pred)
std_err = np.std(y_pred) / np.sqrt(n)
t_value = 1.96  # for a 95% confidence interval and n > 30
confidence_interval = (m - t_value * std_err, m + t_value * std_err)
print("95% confidence interval:", confidence_interval)



