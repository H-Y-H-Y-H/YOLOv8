import numpy as np

dataset_path = "/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/knolling_data_small/"



for i in range(8):

    # first remove all zero data
    # Replace first item to 0, class 0
    # Scale x,y into a ratio
    # Swap x, y --> Y, X
    # Scale W,H for visualization

    # size scale
    raw_d = np.loadtxt(dataset_path + 'labels/train/%012d.txt' % i)
    # cut_id = np.where(raw_d[:,0] == 0)[0][0]

    rm_zero_d = raw_d
    # rm_zero_d[:,0] = 0
    # print(rm_zero_d)
    # rm_zero_d[:,1] /= 0.3
    # rm_zero_d[:,2] += 0.2
    # rm_zero_d[:, 2] /= 0.4
    # print(rm_zero_d)
    # swap = np.copy(rm_zero_d[:,1])
    # rm_zero_d[:,1] = rm_zero_d[:,2]
    # rm_zero_d[:, 2] = swap
    # print(rm_zero_d)
    #
    # rm_zero_d[:,3:5] *= 3
    #
    # rm_zero_d[:, 5] = (rm_zero_d[:, 5] + 1) / 2
    # rm_zero_d[:, 6] = (rm_zero_d[:, 6] + 1) / 2
    rm_zero_d[:, 5] = rm_zero_d[:, 5]
    rm_zero_d[:, 6] = 0.9


    # break
    np.savetxt(dataset_path+'labels/train/%012d.txt'%i, rm_zero_d, fmt='%f')



# import os

# for i in range(10):
    # os.rename(dataset_path+'raw/img%d.png'%i, dataset_path+'images/%012d.png'%i)
