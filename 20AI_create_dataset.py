import os, cv2, glob
from sklearn.model_selection import train_test_split
import numpy as np
import random
#將原始圖片resize後存在images 串列, 標籤存在labels串列
images = []
labels = []
dict_labels = {'Fairplane':0, 'Fautomobile':1, 'Fbird':2, 'Fcat':3, 'Fdeer':4, 'Fdog':5, 'Ffrog':6, 'Fhorse':7, 'Fship': 8, 'Ftruck':9, 'Tairplane':10, 'Tautomobile':11, 'Tbird':12, 'Tcat':13, 'Tdeer':14, 'Tdog':15, 'Tfrog':16, 'Thorse':17, 'Tship': 18, 'Ttruck':19}
size = (80, 80)
folders = glob.glob(r"D:\文書處理David\AI_or_not\archive (1)\test\FAKE")
#C:\Users\david\Desktop\Python\detectAI\archive (1)
print(folders)
step = 1
"D:\文書處理David\AI_or_not\archive (1)\train"
for folders in glob.glob(r"C:\Users\david\Desktop\Python\detectAI\archive (1)\train/*"):
    print(folders, 'reading...')
    for filename in os.listdir(folders):          
        #print(filename)
        n = filename.find('(')
        if n == -1:
            if step>=50000:
                label = 10
            else:
                label = 0
        else:
            if step>=50000:
                label = int(filename[n+1:-5:])-1+10
            else:
                label = int(filename[n+1:-5:])-1
        #print(label)
        #label = folders.split('\\')[-1]
        print(os.path.join(folders, filename))
        try:
            img = cv2.imread(os.path.join(folders, filename))#用os來固定path
            print(img)
            if img is not None:
                img = cv2.resize(img, dsize = size)#dsize是目標大小(size(80, 80))
                images.append(img)
                labels.append(label)#label是key(0-19)
        except:
            print(os.path.join(folders, filename), '無法讀取!')
            pass
        step+=1
#print(images, labels)

print(len(images), len(labels))

test_images = []
test_labels = []
step = 1
"D:\文書處理David\AI_or_not\archive (1)\test"
for folders in glob.glob(r"C:\Users\david\Desktop\Python\detectAI\archive (1)\test/*"):
    print(folders, 'reading...')
    for filename in os.listdir(folders):          
        #print(filename)
        n = filename.find('(')
        if n == -1:
            if step>=10000:
                label = 10
            else:
                label = 0
        else:
            if step>=10000:
                label = int(filename[n+1:-5:])-1+10
            else:
                label = int(filename[n+1:-5:])-1
        #print(label)
        #label = folders.split('\\')[-1]
        print(os.path.join(folders, filename))
        try:
            img = cv2.imread(os.path.join(folders, filename))#用os來固定path
            print(img)
            if img is not None:
                img = cv2.resize(img, dsize = size)#dsize是目標大小(size(80, 80))
                test_images.append(img)
                test_labels.append(label)#label是key(0-19)
        except:
            print(os.path.join(folders, filename), '無法讀取!')
            pass
        step+=1
#print(images, labels)
print(len(test_images), len(test_labels))

"""
filename = glob.glob(folders[0])
filename2 = glob.glob(folders[1])
filename2 = os.listdir(folders)

print(filename)
print(filename2)"""


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
print(test_feature.shape, test_feature)
print('start saving...')
data_path = 'CIFake_Datatest/'
if not os.path.exists(data_path): 
    os.makedirs(data_path)       #如果沒有資料夾就創一個
np.save(data_path+'train_feature.npy', train_feature)
np.save(data_path+'train_label.npy', train_label)
np.save(data_path+'test_feature.npy', test_feature)
np.save(data_path+'test_label.npy', test_label)



#建立訓練資料和測試資料，包括訓練\測試特徵集，訓練\測試標籤集
"""print(labels[:10])
train_feature, test_feature, train_label, test_label = train_test_split(
    images,
    labels,
    test_size = 0.2,
    train_size= 0.8,
    random_state = 42
)
train_feature = np.array(train_feature)
train_label= np.array(train_label)
test_feature = np.array(test_feature)
test_label = np.array(test_label)

print(len(train_feature), len(test_feature))
print(train_feature.shape, train_label.shape)
print(test_feature.shape, test_feature)

print('start saving...')
data_path = 'CIFake_Datatest/'
if not os.path.exists(data_path):
    os.makedirs(data_path)       #如果沒有資料夾就創一個
np.save(data_path+'train_feature.npy', train_feature)
np.save(data_path+'train_label.npy', train_label)
np.save(data_path+'test_feature.npy', test_feature)
np.save(data_path+'test_label.npy', test_label)"""