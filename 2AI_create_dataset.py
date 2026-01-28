import os, cv2, glob
from sklearn.model_selection import train_test_split
import numpy as np
import random
#將原始圖片resize後存在images 串列, 標籤存在labels串列
images = []
labels = []
test_images = []
test_labels = []
dict_labels = {'Fake':0, 'True':1}
size = (80, 80)
#C:\Users\david\Desktop\Python\detectAI\archive (1)
step = 1
link = [r"C:\Users\david\Desktop\Python\detectAI\archive (1)\train/*", r"C:\Users\david\Desktop\Python\detectAI\archive (1)\test/*"]
num = [50000, 10000]
file = [[images, labels], [test_images, test_labels]]
for i in range(2):
    step = 1
    for folders in glob.glob(link[i]):
        print(folders, 'reading...')
        for filename in os.listdir(folders):          
            if step>=num[i]:
                label = 1
            else:
                label = 0

            #print(label)
            #label = folders.split('\\')[-1]
            print(os.path.join(folders, filename))
            try:
                img = cv2.imread(os.path.join(folders, filename))#用os來固定path
                print(img)
                if img is not None:
                    img = cv2.resize(img, dsize = size)#dsize是目標大小(size(80, 80))
                    file[i][0].append(img)
                    file[i][1].append(label)#label是key(0, 1)
            except:
                print(os.path.join(folders, filename), '無法讀取!')
                pass
            step+=1
    #print(images, labels)

    print(len(file[i][0]), len(file[i][1]))


# Combine the images and labels into pairs
image_label_pairs = list(zip(images, labels))
test_image_label_pairs = list(zip(test_images, test_labels))
# Shuffle the pairs
random.shuffle(image_label_pairs)
random.shuffle(test_image_label_pairs)
# Split the shuffled pairs back into images and labels
shuffled_images, shuffled_labels = zip(*image_label_pairs)
shuffled_test_images, shuffled_test_labels = zip(*test_image_label_pairs)
train_feature = np.array(shuffled_images)
train_label= np.array(shuffled_labels)
test_feature = np.array(shuffled_test_images)
test_label = np.array(shuffled_test_labels)

print(len(train_feature), len(test_feature))
print(train_feature.shape, train_label.shape)
print(test_feature.shape, test_label.shape)
print('start saving...')
data_path = 'CIFake_Datatest/'
if not os.path.exists(data_path): 
    os.makedirs(data_path)       #如果沒有資料夾就創一個
np.save(data_path+'train_feature2.npy', train_feature)
np.save(data_path+'train_label2.npy', train_label)
np.save(data_path+'test_feature2.npy', test_feature)
np.save(data_path+'test_label2.npy', test_label)
