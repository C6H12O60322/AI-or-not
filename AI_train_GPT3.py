import tensorflow as tf
import numpy as np
import keras.layers as layers
from keras.models import Sequential, Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Conv2DTranspose
from keras.utils import to_categorical

# Define the Mask R-CNN model with VGG16
def build_mask_rcnn_model(input_shape, num_classes):
    # Feature extraction using a pre-trained CNN (VGG19 in this example)
    # Create a Sequential model
    vgg16_model = Sequential()
    # Add the input layer (224x224 RGB image)
    vgg16_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    vgg16_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #GlobalAveragePooling2D
    #conv2d

    # Flatten the feature maps
    vgg16_model.add(Flatten())

    # Add fully connected layers
    vgg16_model.add(Dense(4096, activation='relu'))
    vgg16_model.add(Dense(4096, activation='relu'))
    vgg16_model.add(Dense(1000, activation='softmax'))  # Output layer with 1000 classes (for ImageNet)
    #for layer in vgg16_model.layers[:-3]:
    for layer in vgg16_model.layers:
        layer.trainable = False

    # Region Proposal Network (RPN)
    rpn_conv = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(vgg16_model.layers[-4].output)
    rpn_class = layers.Conv2D(2, (1, 1), activation='softmax')(rpn_conv)
    rpn_bbox = layers.Conv2D(4, (1, 1))(rpn_conv)

    # Object Detection Head (classification and bounding box regression)
    x = layers.GlobalAveragePooling2D()(vgg16_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)

    # Mask Head (for instance segmentation)
    mask_head = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(vgg16_model.layers[-6].output)
    mask_head = layers.Conv2D(num_classes, (1, 1), activation='softmax')(mask_head)

    # Add the following layers
    #conv2d_transpose = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(vgg16_model.layers[-6].output)
    #conv2d_transpose = layers.Conv2D(20, (1, 1))(conv2d_transpose)

    conv2d_1 = layers.Conv2D(20, (1, 1), activation='softmax')(rpn_class)
    conv2d_2 = layers.Conv2D(20, (1, 1))(rpn_bbox)
    conv2d_3 = layers.Conv2D(20, (1, 1))(mask_head)
    # Build the Mask R-CNN model
    #mask_rcnn_model = Model(inputs=vgg16_model.input, outputs=[rpn_class, rpn_bbox, x, mask_head])
    mask_rcnn_model = Model(inputs=vgg16_model.input, outputs=[conv2d_1, conv2d_2, x, conv2d_3])
    for output in mask_rcnn_model.outputs:
        print(output.shape)
    return mask_rcnn_model




# Define input shape and number of classes
input_shape = (80, 80, 3)  
num_classes = 20  
# Load training and validation data 
train_data =  np.load(r'CIFake_Datatest\train_feature.npy')
train_labels =  np.load(r'CIFake_Datatest\train_label.npy')
val_data =  np.load(r'CIFake_Datatest\test_feature.npy')
val_labels =  np.load(r'CIFake_Datatest\test_label.npy')

# Convert integer labels to one-hot encoding
train_labels_one_hot = to_categorical(train_labels, num_classes)
val_labels_one_hot = to_categorical(val_labels, num_classes)

# Build the Mask R-CNN model with VGG16
mask_rcnn_model = build_mask_rcnn_model(input_shape, num_classes)

# Print model summary
mask_rcnn_model.summary()

# 定義損失函數
losses = {
    "conv2d_4": "categorical_crossentropy",
    "conv2d_5": "mse",
    "dense_1": "categorical_crossentropy",
    "conv2d_6": "categorical_crossentropy"
}


# Compile the full model
mask_rcnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss=losses, #losses
                        metrics=['accuracy'])



# Define callbacks for early stopping and model checkpointing
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint_callback = ModelCheckpoint(r"C:\Users\david\Desktop\Python\detectAI\best_mask_rcnn_model.h5", monitor='val_loss', save_best_only=True, verbose=1)



# Train the model with callbacks
history = mask_rcnn_model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                              epochs=10, batch_size= 32,
                              callbacks=[early_stopping_callback, model_checkpoint_callback])

