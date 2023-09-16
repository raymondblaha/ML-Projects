import tensorflow as tf
import sys
from time import perf_counter
import pickle
import ssl

# Here are some constants that worked OK.
# Please try to improve on them
DROPOUT_RATE = 0.03
BATCH_SIZE = 2000
EPOCHS = 100
RELU_LR = 4e-5
SWISH_LR = 4e-5
BATCHNORM_LR = 1e-3
INITIAL_LR = 1e-3
DECAY_RATE_LR = 10 * INITIAL_LR / EPOCHS
ADAMW_DECAY = 0.03

if len(sys.argv) == 1:
    config = "base"
else:
    config = sys.argv[1].strip()

print(f"*** {config} ***")

print("Getting data...", end="", flush=True)

# Accept unverified certificates for fetch
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Get the CIFAR dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Break training data into training and validation
X_train = X_train_full[2000:]/255.0
y_train = y_train_full[2000:]
X_valid = X_train_full[:2000]/255.0
y_valid = y_train_full[:2000]
X_test = X_test/255.0

print("Done")

print(f"Train:{X_train.shape[0]},  Validate:{X_valid.shape[0]},  Test:{X_test.shape[0]}")

# Create a sequential model
model = tf.keras.Sequential()

# Swish activation replace ReLU
# base: relu activate, Nadam
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# swish: base but replace relu with swish
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='swish', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='swish', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# batchnorm: swish but add batch normalization after each full connected layer
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('swish'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='swish', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='swish', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# adamw: schedule but with AdamW optimizer
schedule = tf.keras.optimizers.schedules.ExponentialDecay(INITIAL_LR, decay_steps=1000, decay_rate=0.10, staircase=True)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=schedule, decay=ADAMW_DECAY)
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='swish', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='swish', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#dropout: swish but with dropout
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='swish', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='swish', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='swish'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation='swish'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Make an optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# Convert integer-encoded labels to one-hot encoded labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Compile the model (make sure you get the accuracy metric recorded as part of the history)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
print(model.summary())

print("Starting fit")

start_time = perf_counter()

# Pass the one-hot encoded labels to model.fit()
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_valid, y_valid), verbose=1)

stop_time = perf_counter()

with open(f'history_{config}.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print(history.params)

# Get the loss and accuracy for the test data

result =  model.evaluate(X_test, y_test)
with open("results.csv","a") as out:
    print(f"{config},{stop_time - start_time:.2f},{result[0]:.4f},{100.0 * result[1]:.1f}", file=out)