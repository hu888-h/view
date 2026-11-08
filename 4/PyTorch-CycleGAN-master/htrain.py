#!/usr/bin/python3
# 指定使用 Python 3 解释器
import argparse
# 导入命令行参数解析模块
import itertools
# 导入迭代工具，用于把两个生成器的参数串起来
import torchvision.transforms as transforms
# 导入 torchvision 图像变换模块
from torch.utils.data import DataLoader
# 导入数据加载器
from torch.autograd import Variable
# 导入 Variable，旧版 PyTorch 的自动求导包装
from PIL import Image
# 导入 PIL 图像处理库
import torch
# 导入 PyTorch
# from visdom import Visdom
# 原代码中已注释，导入 Visdom 可视化工具
from models import Generator
# 从 models 模块导入生成器
from models import Discriminator
# 从 models 模块导入判别器
from utils import ReplayBuffer
# 从 utils 导入回放缓冲区
from utils import LambdaLR
# 从 utils 导入自定义学习率调度器
from utils import Logger
# 从 utils 导入日志记录器
from utils import weights_init_normal
# 从 utils 导入正态分布权重初始化函数
from datasets import ImageDataset
# 从 datasets 导入自定义图像数据集类
parser = argparse.ArgumentParser()
# 创建命令行参数解析器
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
# 添加起始 epoch 参数，默认 0
parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs of training')
# 添加训练总轮数参数，默认 40
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
# 添加批次大小参数，默认 1
parser.add_argument('--dataroot', type=str, default='./datasets/horse2zebra/', help='root directory of the dataset')
# 添加数据集根目录参数
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
# 添加初始学习率参数，默认 0.0002
parser.add_argument('--decay_epoch', type=int, default=10, help='epoch to start linearly decaying the learning rate to 0')
# 添加学习率开始线性衰减的 epoch 参数
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
# 添加图像裁剪尺寸参数，默认 256
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
# 添加输入通道数参数，默认 3
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
# 添加输出通道数参数，默认 3
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
# 添加是否使用 GPU 的标志参数
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
# 添加数据加载使用的 CPU 线程数参数
opt = parser.parse_args()
# 解析命令行参数
print(opt)
# 打印解析后的参数
opt.cuda = opt.cuda or torch.cuda.is_available()
# 如果命令行指定了 cuda 或系统有可用 GPU，则使用 GPU
if opt.cuda:
# 如果使用 GPU
    print("Using GPU computation.")
    # 打印使用 GPU 计算
else:
# 否则
    print("Using CPU computation.")
    # 打印使用 CPU 计算
netG_A2B = Generator(opt.input_nc, opt.output_nc)
# 创建 A 到 B 的生成器
netG_B2A = Generator(opt.output_nc, opt.input_nc)
# 创建 B 到 A 的生成器
netD_A = Discriminator(opt.input_nc)
# 创建 A 域判别器
netD_B = Discriminator(opt.output_nc)
# 创建 B 域判别器
if opt.cuda:
# 如果使用 GPU
    netG_A2B.cuda()
    # 将 A2B 生成器移到 GPU
    netG_B2A.cuda()
    # 将 B2A 生成器移到 GPU
    netD_A.cuda()
    # 将 A 判别器移到 GPU
    netD_B.cuda()
    # 将 B 判别器移到 GPU
# `apply()`遍历网络所有层，执行`weights_init_normal`：正态分布初始化网络权重，GAN 训练权重初始化很关键，防止梯度爆炸/消失。
# 原注释：遍历网络所有层，执行正态分布权重初始化
netG_A2B.apply(weights_init_normal)
# 对 A2B 生成器应用权重初始化
netG_B2A.apply(weights_init_normal)
# 对 B2A 生成器应用权重初始化
netD_A.apply(weights_init_normal)
# 对 A 判别器应用权重初始化
netD_B.apply(weights_init_normal)
# 对 B 判别器应用权重初始化
criterion_GAN = torch.nn.MSELoss()
# 定义对抗损失为均方误差损失
criterion_cycle = torch.nn.L1Loss()
# 定义循环一致性损失为 L1 损失
criterion_identity = torch.nn.L1Loss()
# 定义身份损失为 L1 损失
# 鉴别损失，把 B 直接喂给 G_A2B，理想情况下，G_A2B 不应该修改 B 图片；同理 A 喂给 G_B2A，输出应该等于 A
# 原注释：解释身份损失的作用
# 作用：约束生成器对**本域图像不做不必要修改**，防止色彩偏移
# 原注释：进一步说明身份损失作用
# 生成器联合优化器，同时更新`netG_A2B`和`netG_B2A`两个生成器的参数
# 原注释：生成器联合优化器
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
# 创建 Adam 优化器，同时优化两个生成器的参数
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# 创建 A 判别器的 Adam 优化器
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# 创建 B 判别器的 Adam 优化器
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# 为生成器优化器创建学习率调度器
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# 为 A 判别器优化器创建学习率调度器
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# 为 B 判别器优化器创建学习率调度器
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
# 根据是否使用 GPU 选择张量类型
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
# 预分配 A 域输入张量
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# 预分配 B 域输入张量
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
# 创建真实样本标签，值为 1，不需要梯度
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
# 创建假样本标签，值为 0，不需要梯度
# 图像回放缓冲区（ReplayBuffer），保存历史生成的假图，训练判别器时，不从当前 batch 拿假图，而是从缓冲区随机取旧假图
# 原注释：解释回放缓冲区
fake_A_buffer = ReplayBuffer()
# 创建 A 域假图回放缓冲区
fake_B_buffer = ReplayBuffer()
# 创建 B 域假图回放缓冲区
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
# 定义图像预处理：缩放、随机裁剪、随机水平翻转、转张量、归一化
dataset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)
# 创建图像数据集实例
# print(f"Number of samples in dataset: {len(dataset)}")
# 原代码中已注释，打印数据集样本数
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
# 创建数据加载器，打乱数据，使用指定 CPU 线程
logger = Logger(opt.n_epochs, len(dataloader))
# 创建日志记录器
###################################
# 分隔线
if __name__ == '__main__':
# 如果作为主程序运行
    for epoch in range(opt.epoch, opt.n_epochs):
    # 遍历每个 epoch
        for i, batch in enumerate(dataloader):
        # 遍历数据加载器中的每个 batch
            real_A = Variable(input_A.copy_(batch['A']))
            # 将 A 域数据拷贝到预分配张量并包装成 Variable
            real_B = Variable(input_B.copy_(batch['B']))
            # 将 B 域数据拷贝到预分配张量并包装成 Variable
            optimizer_G.zero_grad()
            # 清空生成器梯度
            # 鉴别损失
            # 原注释：鉴别损失
            # G_A2B(B) should equal B if real B is fed
            # 原注释：如果输入真实 B，G_A2B 应输出 B
            same_B = netG_A2B(real_B)
            # 将真实 B 输入 A2B 生成器
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # 计算 B 域身份损失并乘以 5
            # G_B2A(A) should equal A if real A is fed
            # 原注释：如果输入真实 A，G_B2A 应输出 A
            same_A = netG_B2A(real_A)
            # 将真实 A 输入 B2A 生成器
            loss_identity_A = criterion_identity(same_A, real_A)*5.0
            # 计算 A 域身份损失并乘以 5
            # 对抗损失
            # 原注释：对抗损失
            fake_B = netG_A2B(real_A)
            # 用 A2B 生成器生成假 B
            pred_fake = netD_B(fake_B)
            # 用 B 判别器判断假 B
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
            # 计算 A2B 对抗损失，希望判别器认为假 B 为真
            fake_A = netG_B2A(real_B)
            # 用 B2A 生成器生成假 A
            pred_fake = netD_A(fake_A)
            # 用 A 判别器判断假 A
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
            # 计算 B2A 对抗损失
            # 循环一致损失
            # 原注释：循环一致损失
            recovered_A = netG_B2A(fake_B)
            # 将假 B 再转回 A
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
            # 计算 A→B→A 循环损失并乘以 10
            recovered_B = netG_A2B(fake_A)
            # 将假 A 再转回 B
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
            # 计算 B→A→B 循环损失并乘以 10
            # 训练生成器的目标：骗过判别器 + 保证循环一致 + 身份不变
            # 原注释：生成器训练目标
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            # 总生成器损失为六项损失之和
            loss_G.backward()
            # 反向传播生成器损失
            optimizer_G.step()
            # 更新生成器参数
            ###################################
            # 分隔线
            optimizer_D_A.zero_grad()
            # 清空 A 判别器梯度
            pred_real = netD_A(real_A)
            # A 判别器判断真实 A
            loss_D_real = criterion_GAN(pred_real, target_real)
            # 计算真实 A 的判别损失
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            # 将假 A 放入回放缓冲区并取出一张假 A
            pred_fake = netD_A(fake_A.detach())
            # A 判别器判断假 A，detach 阻止梯度回传生成器
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # 计算假 A 的判别损失
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            # A 判别器总损失为真实与假损失平均
            loss_D_A.backward()
            # 反向传播 A 判别器损失
            optimizer_D_A.step()
            # 更新 A 判别器参数
            ###################################
            # 分隔线
            optimizer_D_B.zero_grad()
            # 清空 B 判别器梯度
            pred_real = netD_B(real_B)
            # B 判别器判断真实 B
            loss_D_real = criterion_GAN(pred_real, target_real)
            # 计算真实 B 的判别损失
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            # 将假 B 放入回放缓冲区并取出一张假 B
            pred_fake = netD_B(fake_B.detach())
            # B 判别器判断假 B，detach 阻止梯度回传生成器
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # 计算假 B 的判别损失
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            # B 判别器总损失为真实与假损失平均
            loss_D_B.backward()
            # 反向传播 B 判别器损失
            optimizer_D_B.step()
            # 更新 B 判别器参数
            ###################################
            # 分隔线
            logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                        images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
            # 记录损失和图像
        lr_scheduler_G.step()
        # 更新生成器学习率
        lr_scheduler_D_A.step()
        # 更新 A 判别器学习率
        lr_scheduler_D_B.step()
        # 更新 B 判别器学习率
        torch.save(netG_A2B.state_dict(), './models/netG_A2B.pth')
        # 保存 A2B 生成器权重
        torch.save(netG_B2A.state_dict(), './models/netG_B2A.pth')
        # 保存 B2A 生成器权重
        torch.save(netD_A.state_dict(), './models/netD_A.pth')
        # 保存 A 判别器权重
        torch.save(netD_B.state_dict(), './models/netD_B.pth')
        # 保存 B 判别器权重