import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# load in the data
argv = sys.argv[1]
df = pd.read_csv(argv)

# drop the first row
df.drop(df.index[0], inplace=True)

# Define X and y
X = df[['sentiment', 'probability', 'score', 'Open', 'High', 'Low', 'Volume']]
y = df['Close']

# convert the probability column to a list of values
X['probability'] = X['probability'].apply(lambda x: str(x).strip('[]').split(',') if isinstance(x, str) else x)

# convert each value in the list to a float
X['probability'] = X['probability'].apply(lambda x: [float(i) for i in x] if isinstance(x, list) else x)

# split the list into separate columns
X[['prob1', 'prob2', 'prob3', 'prob4', 'prob5']] = pd.DataFrame(X.probability.tolist(), index=X.index)

# drop the original probability column
X.drop('probability', axis=1, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert the probability columns in X_test to float
X_test[['prob1', 'prob2', 'prob3', 'prob4', 'prob5']] = X_test[['prob1', 'prob2', 'prob3', 'prob4', 'prob5']].astype('float32')

# convert the probability columns in X_train to float
X_train[['prob1', 'prob2', 'prob3', 'prob4', 'prob5']] = X_train[['prob1', 'prob2', 'prob3', 'prob4', 'prob5']].astype('float32')

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Create the model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(1, 11)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))


# Fit the model
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')
model.fit(X_train, y_train, epochs=600, batch_size=128, verbose=4)



# Make predictions
y_pred = model.predict(X_test)

# save the model to a file
model.save('LSTM.h5')

# Print tomorrow's predicted price + the RMSE 
print('Tomorrow\'s predicted price: ', y_pred[-1] + np.sqrt(mean_squared_error(y_test, y_pred)))

# print the RMSE
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))


# Plot the predictions
plt.plot(y_test.values, color='red', label='Real Stock Price')
plt.plot(y_pred, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
plt.savefig('LSTM.png')



