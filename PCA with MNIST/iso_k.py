from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
import numpy as np
from time import perf_counter
import util
import pickle
import random
import sys

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

# IsoMap
k = 41

start = perf_counter()

# Take a random sample of 8000 data points
X_train_sample = random.sample(list(X_train), 8000)

# Use Isomap to reduce the training data to k dimensions
isomap = Isomap(n_components=k)
isomap.fit(X_train_sample)

X_train_isomap = isomap.transform(X_train)

# transform all 60,000 data points using the isomap
X_test_isomap = isomap.transform(X_test)

# Train the random forest classifier (with the same hyperparameters as before) on those
rf = RandomForestClassifier(n_estimators=10, max_depth=10)
rf.fit(X_train_isomap, y_train)

end = perf_counter()

# use the test data to see the accuracy of the system
y_pred = rf.predict(X_test_isomap)
accuracy = accuracy_score(y_test, y_pred)

memory_used = sys.getsizeof(X_train) + sys.getsizeof(X_test) + sys.getsizeof(y_train) + sys.getsizeof(y_test) + sys.getsizeof(X_train_sample) + sys.getsizeof(X_train_isomap) + sys.getsizeof(X_test_isomap) + sys.getsizeof(rf) + sys.getsizeof(isomap)

print("Accuracy: ", accuracy)
print("Time taken: ", end-start, "s")
print("Memory used: ", memory_used, "bytes")


# Trying out all 15 values of k from 784 to 34

# 784
# Accuracy:  0.8612
# Time taken:  120.35566983302124 s

# 627
# Accuracy:  0.8741
# Time taken:  128.85799887496978 s

# 501 
# Accuracy:  0.8745
# Time taken:  113.38887516700197 s
# Memory used:  719664904 bytes

# 401
# Accuracy:  0.8735
# Time taken:  114.15764379210304 s
# Memory used:  663664904 bytes

# 321
# Accuracy:  0.8781
# Time taken:  133.7746604999993 s
# Memory used:  618864904 bytes

# 256
# Accuracy:  0.8872
# Time taken:  125.63347420806531 s
# Memory used:  582464904 bytes

# 205
# Accuracy:  0.8829
# Time taken:  104.28614383307286 s
# Memory used:  553904904 bytes

# 164
# Accuracy:  0.881
# Time taken:  100.33863275009207 s
# Memory used:  530944904 bytes

# # 131
# Accuracy:  0.8833
# Time taken:  106.24003924999852 s
# Memory used:  512464904 bytes

# 105
# Accuracy:  0.8902
# Time taken:  115.36314058303833 s
# Memory used:  497904904 bytes

# 84
# Accuracy:  0.8898
# Time taken:  107.94852312502917 s
# Memory used:  486144904 bytes

# 67
# Accuracy:  0.888
# Time taken:  128.26023379107937 s
# Memory used:  476624904 bytes

# 53
# Accuracy:  0.8894
# Time taken:  142.11743616696913 s
# Memory used:  468784904 bytes

# 43
# Accuracy:  0.8932
# Time taken:  155.2025082919281 s
# Memory used:  463184904 bytes

# 34
# Accuracy:  0.8913
# Time taken:  148.09082745795604 s
# Memory used:  458144904 bytes
