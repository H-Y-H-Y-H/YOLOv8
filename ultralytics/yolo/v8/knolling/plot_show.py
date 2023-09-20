import matplotlib.pyplot as plt
import numpy
import numpy as np
import csv

# test_name = 'yolo_all_random/'
# data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_all_random/'
#
# loss_mean_data = np.loadtxt(data_root + 'pred/results_' + test_name + 'mean.txt')
# loss_mean_data = np.delete(loss_mean_data, -1)
# loss_std_data = np.loadtxt(data_root + 'pred/results_' + test_name + 'std.txt')
# loss_std_data = np.delete(loss_std_data, -1)
#
# print(len(loss_mean_data))
#
# print(len(np.where(0.000005 < loss_mean_data)[0]))
#
# x = np.arange(len(loss_mean_data))
#
# plt.hist(loss_mean_data, range=(0.000005, 0.01))
# plt.xlabel('MSE loss')
# plt.ylabel('Number')
# plt.title('Statistical graph of results with large errors')
# plt.legend()
# plt.show()
# plt.show()
#
# print('loss max', np.max(loss_mean_data))
# print('loss min', np.min(loss_mean_data))
# print('loss mean', np.mean(loss_mean_data))
#
# print('loss std', np.mean(loss_std_data))

def get_data(data_path):
    birth_data = []
    with open(data_path) as f:
        data = csv.reader(f)
        birth_header = next(data)
        print(birth_header)
        for row in data:  # 将csv 文件中的数据保存到birth_data中
            birth_data.append(row)
        data = np.array([[float(x) for x in row] for row in birth_data])
    return data

data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/runs/pose/'

data_path = data_root + 'train_pile_overlap_626/results.csv'
data = get_data(data_path)
train_box_loss_all_random = data[:, 1]

data_path = data_root + 'train_fix_light/results.csv'
data = get_data(data_path)
train_box_loss_fix_light = data[:, 1]

data_path = data_root + 'train_fix_shadow/results.csv'
data = get_data(data_path)
train_box_loss_fix_shadow = data[:, 1]

data_path = data_root + 'train_fix_background/results.csv'
data = get_data(data_path)
train_box_loss_fix_background = data[:, 1]

data_path = data_root + 'train_all_fix/results.csv'
data = get_data(data_path)
train_box_loss_all_fix = data[:, 1]


data_path = data_root + 'train_pile_overlap_626/results.csv'
data = get_data(data_path)
train_pose_loss_all_random = data[:, 2]

data_path = data_root + 'train_fix_light/results.csv'
data = get_data(data_path)
train_pose_loss_fix_light = data[:, 2]

data_path = data_root + 'train_fix_shadow/results.csv'
data = get_data(data_path)
train_pose_loss_fix_shadow = data[:, 2]

data_path = data_root + 'train_fix_background/results.csv'
data = get_data(data_path)
train_pose_loss_fix_background = data[:, 2]

data_path = data_root + 'train_all_fix/results.csv'
data = get_data(data_path)
train_pose_loss_all_fix = data[:, 2]




# train_pose_loss = data[:, 2]
# val_box_loss = data[:, 14]
# val_pose_loss = data[:, 15]

num_data = 150
x = np.arange(num_data)
plt.plot(x, train_box_loss_all_random, label='All random')
# plt.plot(x, train_box_loss_fix_light, label='Fix light')
# plt.plot(x, train_box_loss_fix_shadow, label='Fix shadow')
# plt.plot(x, train_box_loss_fix_background, label='Fix background')
# plt.plot(x, train_box_loss_all_fix, label='All fixed')
plt.plot(x, train_pose_loss_all_random, label='All random')
# plt.plot(x, train_pose_loss_fix_light, label='Fix light')
# plt.plot(x, train_pose_loss_fix_shadow, label='Fix shadow')
# plt.plot(x, train_pose_loss_fix_background, label='Fix background')
# plt.plot(x, train_pose_loss_all_fix, label='All fixed')

plt.xlabel('Epoch')
plt.ylabel('Key point loss')
# plt.ylabel('Box loss')
plt.yscale('log')

plt.title("Comparison of the key point loss of five experiments")
plt.legend()
plt.show()
