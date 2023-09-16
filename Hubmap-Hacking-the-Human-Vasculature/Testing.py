import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def postprocess_predictions(pred_mask, kernel_size=3):
    # We need to binarize the mask by selecting a proper threshold.
    # You may want to tune the threshold value (0.5) depending on your case.
    pred_mask_binarized = pred_mask > 0.5
    pred_mask_binarized = pred_mask_binarized.astype(np.uint8)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply dilation followed by erosion.
    # This is also known as closing and can help to close small holes inside the mask.
    pred_mask_postprocessed = cv2.morphologyEx(pred_mask_binarized, cv2.MORPH_CLOSE, kernel)

    return pred_mask_postprocessed

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define the path to the test image
test_image_path = '//Users/raymondblahajr/Desktop/Kaggle_HuBMAP/hubmap-hacking-the-human-vasculature/test/72e40acccadf.tif'

# Check if the file exists
if not os.path.isfile(test_image_path):
    print(f"Error: Test image file '{test_image_path}' not found.")
else:
    print(f"Test image path: {test_image_path}")

# Read and process the test image
image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 512))
image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size

# Make predictions on the test image
predictions = model.predict(image)

# Apply threshold to get binary mask
threshold = 0.4
labels = (predictions > threshold).astype(np.uint8)

# Visualize the results
plt.figure(figsize=(15, 15))

plt.subplot(1, 3, 1)
plt.imshow(image[0])
plt.axis('off')
plt.title('Test image')

plt.subplot(1, 3, 2)
plt.imshow(predictions[0, ..., 0], cmap='gray')
plt.axis('off')
plt.title('Raw Predictions')

plt.subplot(1, 3, 3)
plt.imshow(labels[0, ..., 0], cmap='gray')
plt.axis('off')
plt.title('Binary Mask After Thresholding')

plt.show()
plt.savefig('test.png')

# print the predictions
print(predictions)
print(labels)
