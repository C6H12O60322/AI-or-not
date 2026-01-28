import tensorflow as tf
import numpy as np
import keras.layers as layers
from keras.models import Sequential, Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Conv2DTranspose
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator

# Define the Mask R-CNN model with VGG16
def build_mask_rcnn_model(input_shape, num_classes):
    # Feature extraction using a pre-trained CNN (VGG19 in this example)
    VGG_weight = r'detectAI\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = VGG16(weights=VGG_weight, include_top=False, input_shape=input_shape)
    """for layer in base_model.layers:
        layer.trainable = False"""

    # Region Proposal Network (RPN)
    rpn_conv = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(base_model.layers[-4].output)
    rpn_class = layers.Conv2D(num_classes, (1, 1), activation='softmax')(rpn_conv)
    rpn_bbox = layers.Conv2D(4, (1, 1))(rpn_conv)

    # Object Detection Head (classification and bounding box regression)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)

    # Mask Head (for instance segmentation)
    mask_head = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(base_model.layers[-6].output)
    mask_head = layers.Conv2D(num_classes, (1, 1), activation='softmax')(mask_head)

    # Build the Mask R-CNN model
    mask_rcnn_model = Model(inputs=base_model.input, outputs=[rpn_class, rpn_bbox, x, mask_head])

    """(None, 5, 5, 2)
    (None, 5, 5, 4)
    (None, 1024)
    (None, 20, 20, 20)"""
    for output in mask_rcnn_model.outputs:
        print(output.shape)
    return mask_rcnn_model



# Define input shape and number of classes
input_shape = (80, 80, 3)  
num_classes = 2
# Load training and validation data 
"""train_data =  np.load(r'CIFake_Datatest\train_feature2.npy')
train_labels =  np.load(r'CIFake_Datatest\train_label2.npy')
val_data =  np.load(r'CIFake_Datatest\test_feature2.npy')
val_labels =  np.load(r'CIFake_Datatest\test_label2.npy')"""

val_data = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\test_feature.npy')
val_labels = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\test_label.npy')
train_data = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\train_feature.npy')
train_labels = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\train_label.npy')
# Normalize the image data
train_data_normalize = train_data / 255.0
val_data_normalize = val_data / 255.0

# Convert the binary labels to one-hot encoding
train_label_onehot = to_categorical(train_labels, num_classes)
val_label_onehot = to_categorical(val_labels, num_classes)

# Ensure the data has the correct shape
train_data_normalize = train_data_normalize.reshape((len(train_data), 80, 80, 3))
val_data_normalize = val_data_normalize.reshape((len(val_data), 80, 80, 3))
"""
train_labels_one_hot = np.zeros((len(train_labels), 80, 80, num_classes))
train_labels_one_hot[:, :, :, 0] = 1 - train_labels  # Class 0 is 1 when label is 0
train_labels_one_hot[:, :, :, 1] = train_labels  # Class 1 is equal to the label
val_labels_one_hot = np.zeros((len(train_labels), 80, 80, num_classes))
val_labels_one_hot[:, :, :, 0] = 1 - train_labels  # Class 0 is 1 when label is 0
val_labels_one_hot[:, :, :, 1] = train_labels  # Class 1 is equal to the label
"""
print(train_label_onehot.shape)
print(val_label_onehot.shape)
# Build the Mask R-CNN model with VGG16
mask_rcnn_model = build_mask_rcnn_model(input_shape, num_classes)

# 定義損失函數
losses = {
    "conv2d_1": "binary_crossentropy",
    "conv2d_2": "mse",
    "dense_1": "binary_crossentropy",
    "conv2d_3": "binary_crossentropy"
}


# Compile the full model
mask_rcnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss=losses, #losses
                        metrics=['accuracy'])

# Print model summary
mask_rcnn_model.summary()

# Define callbacks for early stopping and model checkpointing
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint_callback = ModelCheckpoint(r"C:\Users\david\Desktop\Python\detectAI\best_mask_rcnn_model2.h5", monitor='val_loss', save_best_only=True, verbose=1)



# Train the model with callbacks
history = mask_rcnn_model.fit(train_data_normalize, train_label_onehot, validation_data=(val_data_normalize, val_label_onehot),
                              epochs=10, 
                              batch_size= 8,
                              verbose = 1,
                              callbacks=[early_stopping_callback, model_checkpoint_callback])



scores = mask_rcnn_model.evaluate(val_data_normalize, val_label_onehot)

print(len(scores))
print('\n準確率 =', scores[1])
print('loss =', scores[0])

prediction = mask_rcnn_model.predict(val_data_normalize)
print(prediction[0])
prediction = np.argmax(prediction, axis = 1)
print(prediction[0])

mask_rcnn_model.save('AI_mrcnn_model.h5')
print('模型儲存完畢')
mask_rcnn_model.save_weights('AI_mrcnn_model.weight')
print('模型參數儲存完畢')

def show_predictions_images(images, labels, predictions, start_id, num = 10): #圖片 標籤 預測 

    plt.figure().set_size_inches(12, 14)
    if num > 25: num = 25

    dict_label = {0:'Fake', 1:'Real'}
    for i in range(num):
        ax = plt.subplot(5, 5, i+1) #i+1是因為從1開始不是0
        img = cv2.cvtColor(images[start_id], cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        if len(predictions)>0:
            correct = '(O)' if predictions[start_id] == labels[start_id] else '(X)'
            """
            if predictions[start_id] == labels[start_id]:
                correct = '(O)'
            else:
                correct = '(X)'
            """
            title = f'ai = {str(predictions[start_id])} {correct} label = {str(labels[start_id])}'
        else:
            title = f'label = {str(labels[start_id])}'
        ax.set_title(title, fontsize = 12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1 

    plt.show()
show_predictions_images(val_data_normalize, val_label_onehot, prediction, 0)
