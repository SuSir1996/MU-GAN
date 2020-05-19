# coding:utf-8
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.
"""
多属性生成
Entry point for testing AttGAN network with manipulating multiple attributes.
"""

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

# command: sudo CUDA_VISIBLE_DEVICES=0 python3 test_multi_1_0.py --experiment_name 128_shortcut1_inject1_none --test_atts Pale_Skin Male --test_ints 0.5 0.5 --gpu --num_test 50
def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)
    parser.add_argument('--test_atts', dest='test_atts', nargs='+', help='test_atts')                   # 参与生成的特征
    parser.add_argument('--test_ints', dest='test_ints', type=float, nargs='+', help='test_ints')       # 每个特征的强度
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args(args)

args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_atts = args_.test_atts
args.test_ints = args_.test_ints
args.num_test = args_.num_test
args.load_epoch = args_.load_epoch
args.custom_img = args_.custom_img
args.custom_data = args_.custom_data
args.custom_attr = args_.custom_attr
args.gpu = args_.gpu
args.multi_gpu = args_.multi_gpu

print(args)
# 多属性生成模块
# 输入多属性(保证属性名正确，输入个数和所需个数相等)
assert args.test_atts is not None, 'test_atts should be chosen in %s' % (str(args.attrs))
for a in args.test_atts:
    assert a in args.attrs, 'test_atts should be chosen in %s' % (str(args.attrs))

assert len(args.test_ints) == len(args.test_atts), 'the lengths of test_ints and test_atts should be the same!'

if args.custom_img:
    output_path = join('output', args.experiment_name, 'custom_testing_multi_' + str(args.test_atts))
    from data import Custom
    test_dataset = Custom(args.custom_data, args.custom_attr, args.img_size, args.attrs)
else:
    output_path = join('output', args.experiment_name, 'sample_testing_multi_' + str(args.test_atts))
    if args.data == 'CelebA':
        from data import CelebA
        test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)
    if args.data == 'CelebA-HQ':
        from data import CelebA_HQ
        test_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'test', args.attrs)
# 建立生成图象的保存路径
os.makedirs(output_path, exist_ok=True)
# 每次处理一张图像
test_dataloader = data.DataLoader(
    test_dataset, batch_size=1, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
# 如果没有限定处理多少张，那就把整个数据集都做迁移
if args.num_test is None:
    print('Testing images:', len(test_dataset))
else:
    print('Testing images:', min(len(test_dataset), args.num_test))

# 载入AttGAN模型
attgan = AttGAN(args)
# 载入指定节点
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
progressbar = Progressbar()
# 进行验证
attgan.eval()
# 对图片的大循环
for idx, (img_a, att_a) in enumerate(test_dataloader):
    '''
    负责生成多属性人脸图像
    '''
    if args.num_test is not None and idx == args.num_test:
        break
    
    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)
    att_b = att_a.clone()
    
    for a in args.test_atts:
        i = args.attrs.index(a)
        att_b[:, i] = 1 - att_b[:, i]
        att_b = check_attribute_conflict(att_b, args.attrs[i], args.attrs)  # 对属性进行解耦合

    with torch.no_grad():
        samples = [img_a]                                                       # 第一张原图
        att_b_ = (att_b * 2 - 1) * args.thres_int                               # 把0,1 换成 -0.5 0.5
        for a, i in zip(args.test_atts, args.test_ints):                        # 新的标签以及特征强度
            att_b_[..., args.attrs.index(a)] = att_b_[..., args.attrs.index(a)] * i / args.thres_int
        samples.append(attgan.G(img_a, att_b_))
        # samples = torch.cat(samples, dim=3)
        for ii in range(len(samples)):
            name_str = str(idx + 182638) + '_' + str(ii) + '.jpg'
            vutils.save_image(samples[ii], join(output_path, name_str),nrow=1, normalize=True, range=(-1., 1.))
        print('{:s} done!'.format(str(idx+182638)))
        
