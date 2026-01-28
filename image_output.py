import matplotlib.pyplot as plt
import cv2

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
