import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the .npy files containing your images
train_feature = np.load('CIFake_Datatest/train_feature.npy')
test_feature = np.load('CIFake_Datatest/test_feature.npy')

# Check the shape of the loaded images
print("Shape of train_feature:", train_feature.shape)
print("Shape of test_feature:", test_feature.shape)

# Select an example image from the loaded data
example_image = train_feature[0]  # Change the index as needed

# Display the original and resized images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display

plt.subplot(1, 2, 2)
plt.title("Resized Image")
resized_image = cv2.resize(example_image, (80, 80))
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()