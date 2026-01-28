import numpy as np

# 假設您有一個包含 1000 張圖片的數據集，每張圖片的大小是 (80, 80, 3)
#data = np.random.rand(1000, 80, 80, 3)
train_data =  np.load(r'CIFake_Datatest\train_feature.npy')
train_labels =  np.load(r'CIFake_Datatest\train_label.npy')

l = [train_data, train_labels]
# 設置批次大小
batch_size = 32


# 將數據集分成小批次
num_samples = len(l[0])
num_batches = num_samples // batch_size

for batch_idx in range(num_batches):
    # 計算當前小批次的索引範圍
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx + 1) * batch_size

    # 從數據中提取當前小批次
    batch_data = l[0][start_idx:end_idx]
    batch_labels = l[1][start_idx:end_idx]
    #print(batch_data)
    print(batch_labels)