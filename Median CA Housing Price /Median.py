import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import (
    make_column_transformer,
    make_column_selector,
    ColumnTransformer,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from scipy import stats

# Read the CSV into a pandas dataframe
housing = pd.read_csv("housing.csv")
print("\n*** Raw data ***")

# Look at the first 5 rows of data
print(housing.head())

# Take a quick look at the data frame
print(housing.info())

# How many of each category for "ocean_proximity"
print(housing["ocean_proximity"].value_counts())

# Let's see the numerical attributes
print(housing.describe())

print("\n*** Writing histogram ***")
# Make a histogram of the numerical attributes
housing.hist(bins=50, figsize=(20, 15))
plt.savefig("histogram1.png")

# Create a scatter matrix of the median house value, median income, total rooms, housing median age
# Dropped NaN values 
housing = housing.dropna()
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
scatter_matrix(housing[["median_house_value", "median_income", "total_rooms", "housing_median_age"]], ax=ax)
fig.savefig("scatter_matrix2.png")


# Create a new column for the number of rooms per household
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]

# Create a new column for the number of bedrooms per room
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]

# Create a new column for the population per household
housing["population_per_household"] = housing["population"] / housing["households"]

# Create bins for the median income
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

# Look at the income category properties
print(housing["income_cat"].value_counts())

# Create a histogram of the income category
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing["income_cat"].hist(ax=ax)
fig.savefig("income_cat1.png")

# Create a scatterplot of the median income and the median house value
# Dropped NaN values 
housing = housing.dropna()
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, ax=ax)
fig.savefig("scatter3.png")

# Split the data into a training set and a test set
train,test=train_test_split(housing, test_size=0.2)
print('train set size:',len(train),', test set size:',len(test))

housing['income_cat']=np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

# Create a stratified sample based on the income category
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"])

# Remove the income category
drop = ["income_cat"]

for set_ in (strat_train_set, strat_test_set):
    set_.drop(drop, axis=1, inplace=True)
   
    
# Create a copy of the training set
housing = strat_train_set.copy()

# Create a plot that shows median house value by latitude and longitude
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", 
             figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, ax=ax)
plt.legend()
fig.savefig("scatter4.png")

# Create a correlation matrix
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

print("\n*** Writing correlation matrix ***")
print(corr_matrix)

# Feature engineering
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

print("\n*** Writing correlation matrix ***")
print(corr_matrix)

# Cleaning and removing NaN values 
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

print("\n*** Writing imputer statistics ***")
print(imputer.statistics_)

print("\n*** Writing housing num ***")
print(housing_num.median().values)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,) 

# one-hot encoding
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

print("\n*** Writing categories ***")
print(cat_encoder.categories_)

# Transformer class
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# Tranformer pipeline
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# Use pandas dataframe directly into the pipeline
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

# pipeline for both numerical and categorical attributes
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared

print("\n*** Writing housing prepared ***")
print(housing_prepared)

housing_prepared.shape

# Training the Model 
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Testing the model
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("\n*** Predictions ***")
print("Predictions:", lin_reg.predict(some_data_prepared))

print("/label:", list(some_labels))

# Measure the RMSE
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

print ("\n*** RMSE ***")
print(lin_rmse)

# Decision Tree Regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

print ("\n*** RMSE of Tree Regressor***")
print(tree_rmse)

# Cross Validation
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
display_scores(tree_rmse_scores)

print("\n*** Cross Validation of Linear Regression pt.1 ***")
print (tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

print("\n*** Cross Validation of Linear Regression pt.2 ***")
print (lin_rmse_scores)

# Random Forest Regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

print("\n*** Random Forest Regressor ***")
print(forest_rmse)

# validation scores
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

print("\n*** Cross Validation of Random Forest Regressor ***")
print (forest_rmse_scores)

# fine tune the model with a grid search
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

forest_reg = RandomForestRegressor()

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                            scoring='neg_mean_squared_error',
                            return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
    

print("\n*** Grid Search ***")
print(grid_search)

# best hyperparameters
grid_search.best_params_

# Random search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

print("\n*** Random Search ***")
print(rnd_search)

# K-means clustering
from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(housing_prepared)


print("\n*** K-means Clustering ***")
print(y_pred)


# feature importance
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


print("\n*** Feature Importance ***")
print(feature_importances)

# Check with test set
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


print("\n*** Final RMSE ***")
print(final_rmse)

# 95% confidence interval
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                            loc=squared_errors.mean(),
                            scale=stats.sem(squared_errors)))

# save the model
import joblib


# plot the model predictions
plt.scatter(y_test, final_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

plt.savefig("model_predictions.png")


joblib.dump(final_model, "my_california_housing_model.pkl")




