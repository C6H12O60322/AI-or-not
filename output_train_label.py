import matplotlib.pyplot as plt
import numpy as np

# Load the .npy files containing your images
train_feature = np.load('CIFake_Datatest/train_feature.npy')
test_feature = np.load('CIFake_Datatest/test_feature.npy')
train_label = np.load('CIFake_Datatest/train_label.npy')
test_label = np.load('CIFake_Datatest/test_label.npy')

# Check the shape of the loaded images
print("Shape of train_label:", train_label.shape)
print("Shape of test_label:", test_label.shape)

# Plot the first image
plt.imshow(train_feature[0])
plt.title('Image')
plt.show()

# Plot the corresponding label (mask)
plt.imshow(train_label[0])
plt.title('Label (Mask)')
plt.show()