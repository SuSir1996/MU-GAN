# coding:utf-8
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Entry point for testing AttGAN network."""
import numpy as np
import argparse
import json
import os
from os.path import join

import torch
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar
from utils import find_model

'''
/home/omnisky/syk/test_22/AttGAN-PyTorch-master/data/img_align_celeba/
/home/omnisky/syk/test_22/AttGAN-PyTorch-master/data/list_attr_celeba.txt
生成的标签文件：./output/128_shortcut1_inject1_none/my_att_list.txt
生成的图片文件：./output/128_shortcut1_inject1_none/sample_testing

test.py         输出单一特征的图像每次(13个属性) 14张
test_multi.py   输出多特征的图像(在一张生成图像上迁移多个特征)
test_slide..py  输出属性迁移强度渐变

输入的参数
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--experiment_name 128_shortcut1_inject1_none \
--test_int 1.0 \
--gpu
'''
# comand: sudo CUDA_VISIBLE_DEVICES=0 python3 test_2.py --experiment_name 128_shortcut1_inject1_none --test_int 1.0 --gpu --load_epoch 107

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)         # 实验名
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)             # 属性强度
    parser.add_argument('--num_test', dest='num_test', type=int)                            # 用来测试的张数
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')      # 载入节点的epoch数，默认最后一个
    parser.add_argument('--custom_img', action='store_true')                                # 是否使用自定义数据集
    parser.add_argument('--custom_data', type=str, default='./data/custom')                 # 目标数据集
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')   # 目标数据集标签
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args(args)

args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:                  # 读取实验设置，路径 ./output/128_shortcut1_inject1_none/setting.txt
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))                      # 载入实验参数

args.test_int = args_.test_int                                                              # 特征强度
args.num_test = args_.num_test                                                              # 用来测试的张数
args.gpu = args_.gpu                                                                        # 使用GPU
args.load_epoch = args_.load_epoch                                                          # 载入节点的epoch
args.custom_img = args_.custom_img                                                          # 是否使用自定义数据集
args.custom_data = args_.custom_data                                                        # 目标数据集
args.custom_attr = args_.custom_attr                                                        # 目标的属性
args.n_attrs = len(args.attrs)                                                              # 目标属性的数量
args.betas = (args.beta1, args.beta2)                                                       # adam函数的连个参数

print(args)

output_path = join('output', args.experiment_name, 'sample_testing')                    # 保存生成图像的路径 output/128_shortcut1_inject1_none/sample_testing
from data import CelebA
test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)

# 建立生成图象的保存路径
os.makedirs(output_path, exist_ok=True)                                                     # 建立生成图像的路径 output/128_shortcut1_inject1_none/sample_testing
# 每次处理一张图像
test_dataloader = data.DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, drop_last=False)

# 如果没有限定处理多少张，那就把整个数据集都做迁移
if args.num_test is None:
    print('Testing images:', len(test_dataset))
else:
    print('Testing images:', min(len(test_dataset), args.num_test))

# 载入AttGAN模型
attgan = AttGAN(args)
# 载入指定节点
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))    # 载入指定节点 output/128_shortcut1_inject1_none/checkpoint/
progressbar = Progressbar()
# 进行验证
attgan.eval()
# 对图片的大循环
for idx, (img_a, att_a) in enumerate(test_dataloader):
    '''
    idx:            图像的索引
    img_a:          图像
    att_a:          标签
    原始标签        label_a
    1.文件名        '{:06d}.jpg'.format(idx + 182638)       name_array[i]
    2.生成标签      att_c_list[i]                           att_b_list[i]
    3.生成的图片    samples[i]                              samples[i]
    '''
    # 如果运行到头了，全部生成结束，就跳出循环
    if args.num_test is not None and idx == args.num_test:
        break
    # 图片名称列表
    name_array = []
    for itt in range(13):
        name_i = str(idx + 182638) + '_' + str(itt) + '.jpg'
        name_array.append(name_i)
    
    img_a = img_a.cuda() if args.gpu else img_a
    # 原始标签的 int 备份
    label_a = att_a.clone()
    # label_a [1,13] int64
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)
    # att_a 已经是 0,1序列 且为float
    # att_c_list 作为标签文件
    att_c_list = []             # 13个迁移标签
    for i in range(args.n_attrs):
        tmp = label_a
        tmp[:, i] = 1 - tmp[:, i]   # 取反操作
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        # 还原为[-1,1]的原始标签形式
        tmp = tmp*2-1
        att_c_list.append(tmp)      # 把生成的 int64 标签存起来
    # 这边都是浮点tensor形式
    att_b_list = []        # 原版一共有15张图像，1张原图，1张重建图像，13张迁移图像；标签有14个，1个原标签(重建)，13个迁移标签
    for i in range(args.n_attrs):
        tmp = att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]   # 取反操作
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)
        # att_b_list 是一个放tensor的列表
        # print(att_b_list)
        # print(att_b_list[10].cpu().numpy())
    # 生成标签文件
    with open('output/128_shortcut1_inject1_none/my_att_list3.txt',"a+") as f:      # 生成图像的标签索引 output/128_shortcut1_inject1_none/my_att_list.txt
        for j in range(13):                                                         # j 13张生成图像
            gen_att = name_array[j] + ' '                                           # 第j张图像的名字
            att_j = np.squeeze(att_c_list[j].cpu().numpy())                         # 第j张图像对应的label
            tar_str = []
            for jj in range(13):                                                    # 对第j张图片对应的 label 中的13个属性进行迭代
                e_jj = str(att_j[jj])  #取出第j个特征向量中的第jj个特征
                if jj <= 11:
                    gen_att = gen_att + e_jj + ' '                                  # 前12个属性之间加空格
                else:
                    gen_att = gen_att + e_jj + '\n'                                 # 最后的属性后面加换行符
            f.write(gen_att)
    print('done!')

