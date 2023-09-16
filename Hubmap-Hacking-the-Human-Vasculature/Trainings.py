import os
import json
import cv2
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, BatchNormalization, Dense, Flatten, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import preprocessing

# Define the image size
image_size = 512
input_size = (image_size, image_size, 3)  # Add the number of channels

# Load CSV metadata
tile_meta = pd.read_csv('/Users/raymondblahajr/Desktop/Kaggle_HuBMAP/hubmap-hacking-the-human-vasculature/tile_meta.csv')
wsi_meta = pd.read_csv('/Users/raymondblahajr/Desktop/Kaggle_HuBMAP/hubmap-hacking-the-human-vasculature/wsi_meta.csv')

# Merge tile_meta and wsi_meta based on the 'source_wsi' field
merged_meta = pd.merge(tile_meta, wsi_meta, on='source_wsi')

# Define the path to the JSONL annotations
jsonl_path = '/Users/raymondblahajr/Desktop/Kaggle_HuBMAP/hubmap-hacking-the-human-vasculature/polygons.jsonl'

# Load JSONL annotations into a DataFrame
with open(jsonl_path, 'r') as f:
    annotations = [json.loads(line) for line in f]
annotations_df = pd.DataFrame(annotations)

# Merge annotations with metadata based on the 'id' field
merged_meta = pd.merge(merged_meta, annotations_df, on='id')

# Load images into memory
image_dir = '/Users/raymondblahajr/Desktop/Kaggle_HuBMAP/hubmap-hacking-the-human-vasculature/train'
merged_meta['image'] = [np.array(Image.open(os.path.join(image_dir, f'{id_}.tif'))) for id_ in merged_meta['id']]


def create_mask(image, annotations):
    # Create an empty mask
    mask = Image.new('L', (image.width, image.height), 0)
    
    # Create a draw object
    draw = ImageDraw.Draw(mask)
    
    # Draw each annotation on the mask
    for annotation in annotations:
        # Convert coordinates to tuples for PIL
        points = [(point[0], point[1]) for point in annotation['coordinates'][0]]
        draw.polygon(points, outline=1, fill=1)
        
            # Convert the mask to a numpy array and return it
    return np.array(mask)



def preprocess_masks(masks):
    # Convert the masks to binary classification format
    return np.array([1 if np.any(mask) else 0 for mask in masks], dtype=np.float32)


# Create the mask images and assign them directly to the 'mask' column
merged_meta['mask'] = [create_mask(Image.fromarray(row['image']), row['annotations']) for _, row in merged_meta.iterrows()]

# Convert the mask images to NumPy array
merged_meta['mask'] = np.array(merged_meta['mask'])

# Split the data into training and validation sets
train_df, val_df = train_test_split(merged_meta, test_size=0.2, random_state=42)


# Define LabelEncoder
le = preprocessing.LabelEncoder()

# Convert sex and race fields to numeric values
train_df['sex'] = le.fit_transform(train_df['sex'])
train_df['race'] = le.fit_transform(train_df['race'])

val_df['sex'] = le.fit_transform(val_df['sex'])
val_df['race'] = le.fit_transform(val_df['race'])

# Extract metadata fields for each image
metadata_fields = ['age', 'sex', 'race', 'height', 'weight', 'bmi']
X_train_metadata = train_df[metadata_fields].values.astype('float32')
X_val_metadata = val_df[metadata_fields].values.astype('float32')


# Convert boolean masks to binary 0 and 1
y_train = preprocess_masks(train_df['mask'])
y_val = preprocess_masks(val_df['mask'])

# Convert boolean masks to binary 0 and 1 tensors
y_train = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(y_val)

# Normalize the image data
X_train_image = train_df['image'].values / 255.0
X_val_image = val_df['image'].values / 255.0

# Convert image data to tensors
X_train_image = tf.convert_to_tensor(X_train_image.tolist(), dtype=tf.float32)
X_val_image = tf.convert_to_tensor(X_val_image.tolist(), dtype=tf.float32)


# U-Net model
def create_unet_model(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    

    # Update the final output layer with the same dimensions as the mask shape
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  

    return Model(inputs=[inputs], outputs=[outputs])


# Metadata model
def create_metadata_model(input_shape):
    inputs = Input(input_shape)
    dense1 = Dense(32, activation='relu')(inputs)
    dense2 = Dense(64, activation='relu')(dense1)
    return Model(inputs=[inputs], outputs=[dense2])


def create_combined_model(image_shape, metadata_shape, mask_shape):
    unet_model = create_unet_model(image_shape)
    meta_model = create_metadata_model(metadata_shape)

    # Flatten the output of UNet model to 2D before concatenation
    unet_flatten = Flatten()(unet_model.output)

    # Concatenate along the last axis (axis=1 for 2D tensors)
    combined_input = Concatenate(axis=1)([unet_flatten, meta_model.output])

    dense1 = Dense(64, activation='relu')(combined_input)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(128, activation='relu')(drop1)
    drop2 = Dropout(0.2)(dense2)
    
    # Add a reshape layer to make the dense layer compatible with mask_shape
    dense = Dense(np.prod(mask_shape), activation='sigmoid')(drop2)
    reshaped = tf.keras.layers.Reshape(mask_shape)(dense)

    return Model(inputs=[unet_model.input, meta_model.input], outputs=[reshaped])


# Create the combined model
model = create_combined_model(input_size, len(metadata_fields), (image_size, image_size, 1))




metadata_fields = ['age', 'sex', 'race', 'height', 'weight', 'bmi']



def data_generator(df, metadata_fields, batch_size=14):
    while True:
        batch_indices = np.random.choice(a=len(df), size=batch_size, replace=False)
        batch_data = []
        for i in batch_indices:
            row = df.iloc[i]
            image = row['image'].astype(np.uint8)   # Convert to uint8
            image = image.reshape((image_size, image_size, 3))  # Ensure the image has correct dimensions
            mask = create_mask(Image.fromarray(image), row['annotations'])
            mask = mask.reshape((image_size, image_size, 1))  # Reshape mask to (image_size, image_size, 1)
            metadata = row[metadata_fields].values.astype('float32')
            batch_data.append((image / 255.0, metadata, mask))
        X_image, X_metadata, y = zip(*batch_data)
        yield [np.array(X_image), np.array(X_metadata)], np.array(y)



# Create the generators
batch_size = 1
epochs = 10
train_generator = data_generator(train_df, metadata_fields, batch_size)
val_generator = data_generator(val_df, metadata_fields, batch_size)



model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define the callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
                ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)]


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size,
    callbacks=callbacks
)
