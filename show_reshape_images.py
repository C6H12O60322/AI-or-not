import matplotlib.pyplot as plt
import numpy as np
"""test_feature = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\test_feature.npy')
test_label = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\test_label.npy')
train_feature = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\train_feature.npy')
train_label = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\train_label.npy')"""

"""print(test_feature.shape)
print(test_label.shape)
print(train_feature.shape)
print(train_label.shape)"""
#reshape
"""train_feature_vector = train_feature.reshape(len(train_feature), 80, 80, 3).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), 80, 80, 3).astype('float32')
train_feature_normalized =train_feature_vector/255
test_feature_normalized = test_feature_vector/255
print(train_feature_normalized[0])
plt.imshow(train_feature_vector[0].astype('uint8'))
plt.show()"""
#C:\Users\david\Desktop\Python\detectAI\dog.jpg
train_feature = np.load(r'C:\Users\david\Desktop\Python\detectAI\dog.jpg')
train_feature_vector = train_feature.reshape(len(train_feature), 80, 80, 3).astype('float32')
train_feature_normalized =train_feature_vector/255
plt.imshow(train_feature_vector[0].astype('uint8'))
plt.show()