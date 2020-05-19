#coding:utf-8
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""AttGAN, generator, and discriminator."""

import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torchsummary import summary


# This architecture is for images of 128x128
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16  # 1024

class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=0, inject_layers=0, img_size=128):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        
        layers = []
        n_in = 3
        # start
        layers += [Conv2dBlock(3, 64, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn)]                   # 0
        layers += [Self_Attn(64, 'relu')]                                                                                       # 1
        layers += [Conv2dBlock(64, 128, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn)]                 # 2
        layers += [Self_Attn(128, 'relu')]                                                                                      # 3
        layers += [Conv2dBlock(128, 256, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn)]                # 4
        layers += [Conv2dBlock(256, 512, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn)]                # 5
        layers += [Conv2dBlock(512, 1024, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn)]               # 6
        self.enc_layers = nn.ModuleList(layers)
        # end

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        if self.inject_layers >= 1:
            k=13
        else:
            k=0
        # start
        layers += [ConvTranspose2dBlock(1037, 1024, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn)]     # 0
        layers += [Attention_block(F_g=1024,F_l=512,F_int=512)]                                                                 # 1
        layers += [conv_block(ch_in=1536,ch_out=1024)]                                                                          # 2
        layers += [ConvTranspose2dBlock(1024+k, 512, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn)]    # 3
        layers += [Attention_block(F_g=512,F_l=256,F_int=256)]                                                                  # 4
        layers += [conv_block(ch_in=768,ch_out=512)]                                                                            # 5
        layers += [ConvTranspose2dBlock(512, 256, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn)]       # 6
        layers += [Self_Attn(256, 'relu')]                                                                                      # 7                                                                 
        layers += [ConvTranspose2dBlock(256, 128, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn)]       # 8
        layers += [Self_Attn(128, 'relu')]                                                                                      # 9
        layers += [ConvTranspose2dBlock(128, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh')]                   # 10
        self.dec_layers = nn.ModuleList(layers)
        # end
    
    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:   #5
            z = layer(z)
            zs.append(z)
        return zs
    
    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size) # 将特征向量延展为 4*4*13
        z = torch.cat([zs[-1], a_tile], dim=1)                                      # 将特征向量concat进编码器的输出，作为解码器的输入 z 8*8*1037
        for i in range(len(self.dec_layers)):
            if i<=6:
                if i%3 == 0:
                    z = self.dec_layers[i](z)
                elif i%3 == 1:
                    x_d = z
                    x_f = zs[5-(i//3)]                  # 编解码器的对应层数关系
                    decode_l = self.dec_layers[i]
                    att_out = decode_l(g=x_d ,x=x_f)
                    z = torch.cat((att_out,x_d),dim=1)
                elif i%3 == 2:
                    z = self.dec_layers[i](z)
                    if self.inject_layers*3 > i:        # 因为改变了层数，所以此处逻辑改了
                        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size * 2**(i//3+1), self.f_size * 2**(i//3+1))
                        z = torch.cat([z, a_tile], dim=1)
            else:
                z = self.dec_layers[i](z)
        return z
    
    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)

class Discriminators(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2**n_layers
        
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 13, 'none', 'none')
        )
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)
        # 返回值 1.真伪标签(1*1) 2.各属性概率标签(1*13)



import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim


# multilabel_soft_margin_loss = sigmoid + binary_cross_entropy

class AttGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_gp = args.lambda_gp
        
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()      # modle父类自带的方法，用于做min_batch的结算
        if self.gpu: self.G.cuda()
        # summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.D = Discriminators(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        # summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        if self.multi_gpu:  # 并行计算
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
    
    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False
        
        zs_a = self.G(img_a, mode='enc')                                    # 输入真实图片
        img_fake = self.G(zs_a, att_b_, mode='dec')                         # 真实图片高级特征 + 目标属性
        img_recon = self.G(zs_a, att_a_, mode='dec')                        # 真实图片高级特征 + 原属性
        d_fake, dc_fake = self.D(img_fake)                                  # 输出生成图片的 真伪/类别
        
        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()                                        # 1.对抗损失 生成样本的损失
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b)        # 2.分类损失 生产图片经过判别网络D 产生的标签的二元交叉熵损失
        gr_loss = F.l1_loss(img_recon, img_a)                               # 3.图像重建损失
        g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG
    
    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True
        
        img_fake = self.G(img_a, att_b_).detach()                   # 生成图像    
        d_real, dc_real = self.D(img_a)                             # 真实图像经过鉴别器得到的标签
        d_fake, dc_fake = self.D(img_fake)                          # 生成图像经过鉴别器得到的标签
        
        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                      F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)     # 梯度惩罚机制
        dc_loss = F.binary_cross_entropy_with_logits(dc_real, att_a)
        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss
        
        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(), 
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item()
        }
        return errD
    
    def train(self):
        self.G.train()
        self.D.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
    
    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
    
    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)

# 2层不改变特征图大小的卷积 加上BN ReLU
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

# 直接上采样为原来的两倍大小
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),                # 固定模式上采样
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
# ATT_U-Net 的 Attention模块
class Attention_block(nn.Module):
    # 128*128*512
    # F_g,F_l 尺寸相等 都比输出大一圈， F_int通道是他们的一半(512, 512, 256)
    def __init__(self,F_g,F_l,F_int):               # 通道 F_g:大尺寸输入 F_l：前级输入 F_int：他们通道的一半
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(                   # 步长为1的1*1卷积 BN
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )                                       # 输出：Hg*Wg*F_int
        
        self.W_x = nn.Sequential(                   # 步长为1的1*1卷积 BN
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )                                           # 输出：Hg*Wg*F_int

        self.psi = nn.Sequential(                   # 步长为1的1*1卷积 BN
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    def forward(self,g,x):
        # g,x 128*128*512
        g1 = self.W_g(g)                            # g支路输出     128*128*256
        x1 = self.W_x(x)                            # Xl支路输出    128*128*256    
        psi = self.relu(g1+x1)                      # 2路信息相加   128*128*256
        psi = self.psi(psi)                         # output       128*128*1 
        return x*psi                                # 与特征图相乘  128*128*512

# self-attention模块(使用标准的卷积，而不是谱归一化)
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):                                                               # 构造函数
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim                                                                         # 输入通道数
        self.activation = activation                                                                    # 父类里的属性，激活函数？？？
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)   # Q通道输出的通道数为原来的8分之一
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)     # K通道输出的通道数为原来的8分之一
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)      # V通道输出的通道数不变
        self.gamma = nn.Parameter(torch.zeros(1))                                                       # att图的权值参数

        self.softmax  = nn.Softmax(dim=-1)                                                              # softmax后形成att_map
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()                                                          # X : B*C*W*H 获取B,C,W,H的维度信息
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)               # B x (C/8) x (W*H) 将Q卷积后的特征拉长为2维的，B*C*(H*W),后经过转置 B*(H*W)*C
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)                                  # B x (C/8) x (W*H) 将K卷积后的特征拉长为2维的，B*C*(H*W)
        energy =  torch.bmm(proj_query,proj_key)                                                        # transpose check  B*(H*W)*(H*W)
        attention = self.softmax(energy)                                                                # B x (N) x (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)                               # B x C x N 将V卷积后的特征拉长为2维的，B*C*(H*W)

        out = torch.bmm(proj_value,attention.permute(0,2,1) )                                           # 将V与attention_map相乘 (B*C*N)(B*N*N) = B*C*N
        out = out.view(m_batchsize,C,width,height)                                                      # 再还原成原图像 B*C*H*W
        
        out = self.gamma*out + x                                                                        # 计算残差 权重参数为 self.gamma 如果残差为0那就是恒等映射
        return out    #attention                                                                        # 返回 1.attention残差结果 2.N*N的注意力图(有什么意义)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)
