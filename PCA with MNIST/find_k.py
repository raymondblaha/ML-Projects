from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from time import perf_counter
import util
import pickle

(X, y) = util.load_data()

# Break into train and test sets
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]

# convert the data to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# start with 784
k = 784
accuracies = []
ks = []
times = []
explained_variances = []


# loop through the values of k
while k > 0:
    # fit the PCA model
    pca = PCA(n_components=k)
    pca.fit(X_train)
    # transform the data
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # fit the random forest model
    start = perf_counter()
    rf = RandomForestClassifier(n_estimators=10, max_depth=10)
    rf.fit(X_train_pca, y_train)
    end = perf_counter()
    # get the accuracy
    y_pred = rf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    # store the values
    accuracies.append(accuracy)
    ks.append(k)
    times.append(end - start)
    explained_variances.append(sum(pca.explained_variance_ratio_)) 
    # decrease k by 20%
    k = int(k * 0.8)
    
# save the accuracies, ks, times, and explained variances to a pickle file
with open("k.pkl", "wb") as f:
    pickle.dump((accuracies, ks, times, explained_variances), f)
