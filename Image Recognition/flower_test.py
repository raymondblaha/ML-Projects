import tensorflow as tf
import tensorflow_datasets as tfds
import sys 
import sklearn
import matplotlib.pyplot as plt 
import numpy as np

# Create a python file reads in a 244X244 image and returns what flower it is 

# load the model
model = tf.keras.models.load_model('flower_model.h5')

# load in the image
img_path = sys.argv[1]
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Preproces the image
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.keras.applications.xception.preprocess_input(img_array)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Make a prediction
predictions = model.predict(img_array)

# Get the predicted class
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
predicted_class = class_names[np.argmax(predictions)]

# Print the predicted class
print(predicted_class)


