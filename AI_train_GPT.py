import tensorflow as tf
import numpy as np
import keras.layers as layers
from keras.models import Sequential, Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define the Mask R-CNN model with VGG16
def build_mask_rcnn_model(input_shape, num_classes):
    """    # Feature extraction using a pre-trained CNN (VGG16 in this example)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)"""
    #建立Sequential模型(Vgg-16)
    base_model = Sequential()
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding = 'same', input_shape = input_shape, activation = 'relu'))#建立卷積層
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))
    base_model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))#建立池化層
    #base_model.add(Dropout(0.05))
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))#建立卷積層2
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))
    base_model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))#建立池化層2
    #base_model.add(Dropout(0.05))
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))#建立卷積層2
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))
    base_model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))#建立池化層2
    #base_model.add(Dropout(0.05))
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))#建立卷積層2
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))
    base_model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))#建立池化層2
    #base_model.add(Dropout(0.05))
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))#建立卷積層2
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))
    base_model.add(Conv2D(filters = 20, kernel_size = (3, 3), padding= 'same', activation='relu'))
    base_model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))#建立池化層2
    #base_model.add(Dropout(0.05))
    base_model.add(Flatten(input_shape = (2, 2, 1))) #建立平坦層
    base_model.add(Dense(units = 512, activation = 'relu'))#建立隱藏層
    base_model.add(Dense(units = 2, activation = 'softmax'))#建立輸出層
    for layer in base_model.layers:
        layer.trainable = False

    # Region Proposal Network (RPN)
    rpn_conv = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(base_model.layers[-4].output)
    rpn_class = layers.Conv2D(2, (1, 1), activation='softmax')(rpn_conv)
    rpn_bbox = layers.Conv2D(4, (1, 1))(rpn_conv)

    # Object Detection Head (classification and bounding box regression)
    base_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    x = layers.GlobalAveragePooling2D()(base_model.output)
    #x = base_model.output
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)

    # Mask Head (for instance segmentation)
    mask_head = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(base_model.layers[-6].output)
    mask_head = layers.Conv2D(num_classes, (1, 1), activation='softmax')(mask_head)

    # Build the Mask R-CNN model
    mask_rcnn_model = Model(inputs=base_model.input, outputs=[rpn_class, rpn_bbox, x, mask_head])

    return mask_rcnn_model

# Define input shape and number of classes
input_shape = (80, 80, 3)  
num_classes = 2  

# Build the Mask R-CNN model with VGG16
mask_rcnn_model = build_mask_rcnn_model(input_shape, num_classes)

# Compile the full model
mask_rcnn_model.compile(optimizer=Adam(lr=0.0001),
                        loss=['categorical_crossentropy', 'mse', 'categorical_crossentropy', 'categorical_crossentropy'],
                        metrics=['accuracy'])

# Print model summary
mask_rcnn_model.summary()

# Define callbacks for early stopping and model checkpointing
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint_callback = ModelCheckpoint(r"C:\Users\david\Desktop\Python\detectAI\best_mask_rcnn_model.h5", monitor='val_loss', save_best_only=True, verbose=1)

# Load training and validation data 
train_data =  np.load(r'CIFake_Datatest\train_feature.npy')
train_labels =  np.load(r'CIFake_Datatest\train_label.npy')
val_data =  np.load(r'CIFake_Datatest\test_feature.npy')
val_labels =  np.load(r'CIFake_Datatest\test_label.npy')

# Train the model with callbacks
history = mask_rcnn_model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                              epochs=10, batch_size=32,
                              callbacks=[early_stopping_callback, model_checkpoint_callback])


