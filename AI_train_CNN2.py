import tensorflow as tf
import numpy as np
import keras.layers as layers
from keras.models import Sequential, Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,  LearningRateScheduler
from keras.layers import Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Conv2DTranspose
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2

# Check for available GPUs
"""gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        """
#from image_output import show_predictions_images
#CNN VS code\Cat_Dog_Datatest\test_feature.npy
#載入資料
test_feature = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\test_feature.npy')
test_label = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\test_label.npy')
train_feature = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\train_feature.npy')
train_label = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\train_label.npy')

"""print(test_feature.shape)
print(test_label.shape)
print(train_feature.shape)
print(train_label.shape)"""
#reshape
train_feature_vector = train_feature.reshape(len(train_feature), 80, 80, 3).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), 80, 80, 3).astype('float32')
print(train_feature_vector.shape)

#normalize
train_feature_normalized =train_feature_vector/255
test_feature_normalized = test_feature_vector/255
#label to one-hot
train_label_onehot = to_categorical(train_label)
test_label_onehot = to_categorical(test_label)

"""datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转图像范围（0-40度之间）
    width_shift_range=0.2,  # 随机水平平移图像的宽度比例
    height_shift_range=0.2,  # 随机垂直平移图像的高度比例
    shear_range=0.2,  # 随机错切变换角度
    zoom_range=0.2,  # 随机缩放图像范围
    horizontal_flip=True,  # 随机水平翻转图像
    fill_mode='nearest'  # 用于填充新像素的策略
)
augmented_data = datagen.flow(train_feature_normalized, train_label_onehot, batch_size=32)"""

def build_VGG16_model(input_shape, l1_value=0.001, l2_value=0.001):
    # Feature extraction using a pre-trained CNN (VGG19 in this example)
    """VGG_weight = r'detectAI\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model = VGG16(weights=VGG_weight, include_top=True, input_shape=input_shape)"""
    # Initialize the Sequential model
    model = Sequential()

    # First Convolutional Layer
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second Convolutional Layer
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Third Convolutional Layer
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fourth Convolutional Layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fifth Convolutional Layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(2, kernel_regularizer=l1(l1_value), bias_regularizer=l2(l2_value)))  # Two output neurons for binary classification
    model.add(Activation('softmax'))
    return model


input_shape = (80, 80, 3)
num_classes = 2
model = build_VGG16_model(input_shape)
model.summary()
#訓練模型 
learning_rate = 0.0001 #0.001
epochs=50
batch_size=32
l = [learning_rate, epochs, batch_size]
optimizer = Adam(learning_rate = learning_rate)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['binary_accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint_callback = ModelCheckpoint(r"C:\Users\david\Desktop\Python\AI_cnn_model.h5", monitor='val_loss', save_best_only=True, verbose=1)


"""history = model.fit(
    augmented_data,
    validation_data=(test_feature_normalized, test_label_onehot),
    epochs=32,
    batch_size = 100,
    verbose=1
)"""
training_loss = []
testing_loss = []
def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        return lr * 0.1
    else:
        return lr
learning_rate_scheduler = LearningRateScheduler(scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
train_history = model.fit(
    x = train_feature_normalized,
    y = train_label_onehot,
    validation_split=0.2,
    epochs = epochs, #32
    batch_size = batch_size,
    verbose = 1,
    callbacks=[early_stopping_callback, model_checkpoint_callback, learning_rate_scheduler]
    )
#callbacks=[early_stopping_callback, model_checkpoint_callback]

# Append the training and testing loss values to the lists
training_loss += train_history.history['loss']
testing_loss += train_history.history['val_loss']

scores2 = model.evaluate(test_feature_normalized, test_label_onehot)
print('\ntest_accuracy =', scores2[1])
print('test_loss =', scores2[0])
"""
scores1 = model.evaluate(train_feature_normalized, train_label_onehot)
scores2 = model.evaluate(test_feature_normalized, test_label_onehot)
print(len(scores1, scores2))
print('\ntrain_accuracy =', scores1[1])
print('train_loss =', scores1[0])
print('\ntest_accuracy =', scores2[1])
print('test_loss =', scores2[0])
"""
prediction = model.predict(test_feature_normalized)
print(prediction[0])
prediction = np.argmax(prediction, axis = 1)
print(prediction[0])

model.save('AI_cnn_model.h5')
print('模型儲存完畢')
model.save_weights('AI_cnn_model.weight')
print('模型參數儲存完畢')
print(f'lr = {l[0]} epochs = {l[1]} batch_size = {l[2]}')
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
show_predictions_images(test_feature, test_label, prediction, 0)

#del model

# Plot training and testing loss
print(training_loss)
print(testing_loss)
epochs = range(1, len(training_loss) + 1)

plt.plot(epochs, training_loss, 'bo', label='Training Loss')
plt.plot(epochs, testing_loss, 'r', label='Testing Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
