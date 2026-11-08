#!/usr/bin/python3
# 指定使用 Python 3 解释器
import argparse
# 导入命令行参数解析模块
import sys
# 导入系统模块，用于标准输出
import os
# 导入操作系统接口模块
import torchvision.transforms as transforms
# 导入 torchvision 图像变换模块
from torchvision.utils import save_image
# 导入保存图像的工具
from torch.utils.data import DataLoader
# 导入数据加载器
from torch.autograd import Variable
# 导入 Variable
import torch
# 导入 PyTorch
from models import Generator
# 从 models 导入生成器
from datasets import ImageDataset
# 从 datasets 导入图像数据集
parser = argparse.ArgumentParser()
# 创建命令行参数解析器
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
# 添加批次大小参数
parser.add_argument('--dataroot', type=str, default='./datasets/horse2zebra/', help='root directory of the dataset')
# 添加数据集根目录参数
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
# 添加输入通道数参数
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
# 添加输出通道数参数
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
# 添加图像尺寸参数
parser.add_argument('--cuda', default=True, help='use GPU computation')
# 添加是否使用 GPU 参数，默认 True
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# 添加 CPU 线程数参数
parser.add_argument('--generator_A2B', type=str, default='./models/pretrained_netG_A2B.pth', help='A2B generator checkpoint file')
# 添加 A2B 生成器权重路径参数
parser.add_argument('--generator_B2A', type=str, default='./models/pretrained_netG_B2A.pth', help='B2A generator checkpoint file')
# 添加 B2A 生成器权重路径参数
opt = parser.parse_args()
# 解析命令行参数
print(opt)
# 打印参数
opt.cuda = opt.cuda or torch.cuda.is_available()
# 如果指定 cuda 或系统有 GPU，则使用 GPU
if opt.cuda:
# 如果使用 GPU
    print("Using GPU computation.")
    # 打印使用 GPU
else:
# 否则
    print("Using CPU computation.")
    # 打印使用 CPU
netG_A2B = Generator(opt.input_nc, opt.output_nc)
# 创建 A2B 生成器
netG_B2A = Generator(opt.output_nc, opt.input_nc)
# 创建 B2A 生成器
if opt.cuda:
# 如果使用 GPU
    netG_A2B.cuda()
    # 将 A2B 生成器移到 GPU
    netG_B2A.cuda()
    # 将 B2A 生成器移到 GPU
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
# 加载 A2B 生成器权重
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
# 加载 B2A 生成器权重
netG_A2B.eval()
# 设置 A2B 生成器为评估模式
netG_B2A.eval()
# 设置 B2A 生成器为评估模式
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
# 根据是否使用 GPU 选择张量类型
# 动态选择张量类型，GPU 就用 cuda 张量，CPU 就用普通 tensor
# 原注释：动态选择张量类型
# 预分配张量容器，用来存放一个 batch 的 A、B 测试图片，后续用`.copy_()`拷贝数据
# 原注释：预分配张量容器
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
# 预分配 A 域输入张量
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# 预分配 B 域输入张量
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
# 定义测试时的图像变换：转张量、归一化
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
# 创建测试数据加载器，不打乱顺序
###################################
# 分隔线
if __name__ == '__main__':
# 如果作为主程序运行
    if not os.path.exists('./output/A/'):
    # 如果输出目录 A 不存在
        os.makedirs('./output/A/')
        # 创建输出目录 A
    if not os.path.exists('./output/B/'):
    # 如果输出目录 B 不存在
        os.makedirs('./output/B/')
        # 创建输出目录 B
    for i, batch in enumerate(dataloader):
    # 遍历测试数据加载器
        # 将图片拷贝到预分配张量，包装成 Variable
        # 原注释：拷贝图片到张量
        real_A = Variable(input_A.copy_(batch['A']))
        # 将 A 域图片拷贝到张量并包装成 Variable
        real_B = Variable(input_B.copy_(batch['B']))
        # 将 B 域图片拷贝到张量并包装成 Variable
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        # 用 A2B 生成器生成假 B，并将值从 [-1,1] 映射到 [0,1]
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
        # 用 B2A 生成器生成假 A，并映射到 [0,1]
        save_image(fake_A, './output/A/%04d.png' % (i+1))
        # 保存假 A 图像
        save_image(fake_B, './output/B/%04d.png' % (i+1))
        # 保存假 B 图像
        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
        # 在标准输出显示生成进度
    sys.stdout.write('\n')
    # 输出换行