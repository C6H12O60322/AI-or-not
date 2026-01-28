import os, cv2, glob
from sklearn.model_selection import train_test_split
import numpy as np
#將原始圖片resize後存在images 串列, 標籤存在labels串列
images = []
labels = []
dict_labels = {'Fake':0, 'Real':1}
size = (80, 80)
#folders = glob.glob(r'Petimages/*')


link = r"C:\Users\david\Desktop\Python\detectAI\archive (1)\train/*"
"""for folders in glob.glob(link):
    print(folders, 'reading...')
    for filename in os.listdir(folders):
        label = folders.split('\\')[-1]
        #print(label)
        #print(os.path.join(folders, filename))
        try:
            img = cv2.imread(os.path.join(folders, filename))#用os來固定path
            if img is not None:
                img = cv2.resize(img, dsize = size)#dsize是目標大小(size(80, 80))
                images.append(img)
                labels.append(dict_labels[label])#label是key，填入cat or dog，labels存入01
        except:
            print(os.path.join(folders, filename), '無法讀取!')
            pass
print(len(images), len(labels))
print(labels)
"""
#create dataset
step = 1
for folders in glob.glob(link):
    print(folders, 'reading...')
    for filename in os.listdir(folders):          
        #print(filename)
        if step>=2:
            label = 1
        else:
            label = 0
        #print(label)
        #label = folders.split('\\')[-1]
        print(os.path.join(folders, filename))
        try:
            img = cv2.imread(os.path.join(folders, filename))#用os來固定path
            #print(img)
            if img is not None:
                img = cv2.resize(img, dsize = size)#dsize是目標大小(size(80, 80))
                images.append(img)
                labels.append(label)#label是key(0 1)
        except:
            print(os.path.join(folders, filename), '無法讀取!')
            pass
    step+=1

"""filename = glob.glob(folders[0])
filename2 = glob.glob(folders[1])
filename2 = os.listdir(folders)

print(filename)
print(filename2)"""

#建立訓練資料和測試資料，包括訓練\測試特徵集，訓練\測試標籤集
#print(labels[:10])
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
print(test_feature.shape, test_label.shape)

print('start saving...')
#C:\Users\david\Desktop\Python\CIFake_Datatest
data_path = r'Medium_AI_Datatest/'
if not os.path.exists(data_path):
    os.makedirs(data_path)       #如果沒有資料夾就創一個
np.save(data_path+'train_feature.npy', train_feature)
np.save(data_path+'train_label.npy', train_label)
np.save(data_path+'test_feature.npy', test_feature)
np.save(data_path+'test_label.npy', test_label)
