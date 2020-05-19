import numpy as np
attr_path = '/home1/syk/AttGAN-PyTorch-master/my_att_list.txt'
images = np.loadtxt(attr_path, usecols=[0], dtype=np.str)       # 读取图片名 np.loadtxt(attr_path txt文件路径, skiprows=2 跳过前两行, usecols=[0] 只用第0列, dtype=np.str 输出形式为字符)
labels = np.loadtxt(attr_path, usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=np.int)      # 读取属性标签
print(images)
print(labels)
print(len(images))
print(len(labels))
print('example: ', labels[2])
