import tensorflow as tf
import sys
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load the training data
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
dataset_size = info.splits['train'].num_examples
class_names = info.features['label'].names
n_classes = info.features['label'].num_classes

# Split the data into training and validation sets
test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    'tf_flowers',
    split=['train[:10%]', 'train[10%:25%]', 'train[25%:]'],
    as_supervised=True) 

# Batch size and preprocessing
BATCH_SIZE = 32

preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224,crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(BATCH_SIZE).prefetch(1)
vaild_set = valid_set_raw.map(lambda X, y: (preprocess(X), y)).batch(BATCH_SIZE)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(BATCH_SIZE)


# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal', seed=42),
    tf.keras.layers.RandomRotation(0.05, seed=42),
    tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2, seed=42)
])


# Create the base model from the pre-trained model Xception
base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation='softmax')(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False
    
# Compile the model
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_set, validation_data=vaild_set, epochs=5)

# Unfreeze the base model
for layer in base_model.layers[56:]:
    layer.trainable = True
    
# Recompile the model
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9, decay=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_set, validation_data=vaild_set, epochs=5)

# save the model to disk
model.save('flower_model.h5')

# Save some images to read into flower_test.py
for X_batch, y_batch in test_set.take(1):
    for i in range(5):
        tf.keras.preprocessing.image.save_img('image' + str(i) + '.jpg', X_batch[i])
        

