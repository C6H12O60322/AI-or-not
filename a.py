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
    VGG_weight = r'detectAI\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = VGG16(weights=VGG_weight, include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Region Proposal Network (RPN)
    rpn_conv = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(base_model.layers[-4].output)
    rpn_class = layers.Conv2D(20, (1, 1), activation='softmax')(rpn_conv)
    rpn_bbox = layers.Conv2D(4, (1, 1))(rpn_conv)

    # Object Detection Head (classification and bounding box regression)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)

    # Mask Head (for instance segmentation)
    mask_head = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(base_model.layers[-6].output)
    mask_head = layers.Conv2D(num_classes, (1, 1), activation='softmax')(mask_head)

    # Add the following layers
    #conv2d_transpose = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(base_model.layers[-6].output)
    #conv2d_transpose = layers.Conv2D(20, (1, 1))(conv2d_transpose)

    #conv2d_1 = layers.Conv2D(20, (1, 1), activation='softmax')(rpn_class)
    #conv2d_2 = layers.Conv2D(20, (1, 1))(rpn_bbox)
    #conv2d_3 = layers.Conv2D(20, (1, 1))(mask_head)

    # Build the Mask R-CNN model
    mask_rcnn_model = Model(inputs=base_model.input, outputs=[rpn_class, rpn_bbox, x, mask_head])
    #mask_rcnn_model = Model(inputs=base_model.input, outputs=[mask_head, mask_head, x, mask_head])
    #mask_rcnn_model = Model(inputs=base_model.input, outputs=[conv2d_1, conv2d_2, x, conv2d_3])
    """(None, 5, 5, 2)
    (None, 5, 5, 4)
    (None, 1024)
    (None, 20, 20, 20)"""
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


l = [train_data, train_labels]
# 設置批次大小
batch_size = 32


    # 將數據集分成小批次
num_samples = len(l[0])
num_batches = num_samples // batch_size
#Convert integer test label to one-hot encoding
val_labels_one_hot = np.zeros((val_labels.shape[0], 80, 80, num_classes))
for i in range(val_labels.shape[0]):
        val_labels_one_hot[i, :, :, val_labels[i]] = 1

for batch_idx in range(num_batches):
    # 計算當前小批次的索引範圍
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx + 1) * batch_size

    # 從數據中提取當前小批次
    batch_data = l[0][start_idx:end_idx]
    batch_labels = l[1][start_idx:end_idx]
    #print(batch_data)
    #print(batch_labels)

    # Convert integer train labels to one-hot encoding

    #train_labels_one_hot = np.zeros((train_labels.shape[0], 80, 80, num_classes))

    for i in range(batch_labels.shape[0]):
        train_labels_one_hot = np.zeros((i, 80, 80, num_classes))
        train_labels_one_hot[i, :, :, batch_labels[i]] = 1  #不用這麼麻煩變one-hot

    # Build the Mask R-CNN model with VGG16
    mask_rcnn_model = build_mask_rcnn_model(input_shape, num_classes)

    # 定義損失函數
    losses = {
        "conv2d_1": "categorical_crossentropy",
        "conv2d_2": "mse",
        "dense_1": "categorical_crossentropy",
        "conv2d_3": "categorical_crossentropy"
    }


    # Compile the full model
    mask_rcnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                            loss=losses, #losses
                            metrics=['accuracy'])

    # Print model summary
    mask_rcnn_model.summary()

    # Define callbacks for early stopping and model checkpointing
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint_callback = ModelCheckpoint(f"C:\\Users\\david\\Desktop\\Python\\detectAI\\best_mask_rcnn_model{batch_idx}.h5", monitor='val_loss', save_best_only=True, verbose=1)



    # Train the model with callbacks
    history = mask_rcnn_model.fit(train_data, train_labels, validation_data=(val_data, val_labels_one_hot),
                                epochs=10, batch_size= 32,
                                callbacks=[early_stopping_callback, model_checkpoint_callback])

