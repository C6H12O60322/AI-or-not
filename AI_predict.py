import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model

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

#建立測試的特徵集、測試標籤
"""files = glob.glob(r'CNN VS code\try/*')
print(files)

test_f = []
test_l = []
size = (80, 80)
dict_label = {'cat':0, 'dog':1}
for file in files:
    img = cv2.imread(file) #轉成numpy矩陣
    #print(img, img.shape)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    #print(img)
    img = cv2.resize(img, dsize = size)
    test_f.append(img)
    label = file[16:19]
    test_l.append(dict_label[label])

print(len(test_f), test_l)"""
#print(test_l)
"""test_f=list(test_f)
test_l=list(test_l)
test_f = numpy.array(test_f)
test_l = numpy.array(test_l)"""
test_f = np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\test_feature.npy')
test_l= np.load(r'C:\Users\david\Desktop\Python\Smaller_AI_Datatest\test_label.npy')
test_f_v = test_f.reshape(len(test_f), 80, 80, 3).astype('float32')
test_f_normalized = test_f_v/255
#print(test_f_normalized[0])

print('載入模型AI_cnn_model.h5')
model = load_model(r'C:\Users\david\Desktop\Python\AI_cnn_model.h5')
prediction = model.predict(test_f_normalized)
prediction = np.argmax(prediction, axis= 1)
print(prediction)
show_predictions_images(test_f, test_l, prediction, 0, 15)

