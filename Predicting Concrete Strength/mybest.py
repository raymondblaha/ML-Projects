import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Read in data
df = pd.read_csv('Concrete_Data_Yeh.csv')

# Now it is time to split the data into training and testing sets
X = df.drop('csMPa', axis=1)
y = df['csMPa']


# Define the hyperparameters to be tuned Best hyperparameters: {'depth': 7, 'iterations': 1000, 'learning_rate': 0.05}
params = {'depth': 7, 'iterations': 1000, 'learning_rate': 0.05}

# Define the function to perform cross-validation
def run_cv(X, y, params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = CatBoostRegressor(depth=params['depth'], iterations=params['iterations'], learning_rate=params['learning_rate'], random_seed=42)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        rmse_scores.append(mean_squared_error(y_test, y_pred, squared=False))
    
    avg_rmse = sum(rmse_scores) / len(rmse_scores)
    return avg_rmse


# call time 
start = time.time()

avg_rmse = run_cv(X, y, params)

print("Average RMSE:", avg_rmse)
print(f"Time taken: {time.time() - start}")


# permuatation importance
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Define the function to calculate permutation importance
model1 = CatBoostRegressor(depth=params['depth'], iterations=params['iterations'], learning_rate=params['learning_rate'], random_seed=42)

def get_permutation_importance(model1, X, y):
    model1.fit(X, y, verbose=False)
    results = permutation_importance(model1, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return results


# Calculate permutation importance
results = get_permutation_importance(model1, X, y)

# Get the indices that would sort the importances in descending order
sorted_idx = results.importances_mean.argsort()[::-1]

# Plot the results
plt.figure(figsize=(10, 8))
plt.bar(X.columns[sorted_idx], results.importances_mean[sorted_idx])
plt.xticks(rotation=90)
plt.show()

# Print feature ranking
print("Feature ranking:")
for i in sorted_idx:
    print(f"{X.columns[i]}: {results.importances_mean[i]:.3f}")