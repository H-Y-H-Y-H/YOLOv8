import numpy as np

dataset_path = "/Users/yuhang/datasets/knolling_data/"
dataset_path2 = "/Users/yuhang/datasets/coco8-pose/labels"


for i in range(10):
    raw_d = np.loadtxt(dataset_path + 'label_raw/normal_409_%d_train.csv'%i)
    cut_id = np.where(raw_d[:,0] == 0)[0][0]

    rm_zero_d = raw_d[:cut_id]
    rm_zero_d[:,0] = 0

    print(rm_zero_d)

    break


