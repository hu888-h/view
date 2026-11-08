import torch.nn as nn
# 导入 PyTorch 神经网络模块
import torch.nn.functional as F
# 导入 PyTorch 函数式接口
class ResidualBlock(nn.Module):
# 定义残差块类，继承 nn.Module
    def __init__(self, in_features):
    # 初始化函数，参数为输入特征数
        super(ResidualBlock, self).__init__()
        # 调用父类初始化
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]
        # 定义残差块内的卷积序列：反射填充、卷积、实例归一化、ReLU、反射填充、卷积、实例归一化
        self.conv_block = nn.Sequential(*conv_block)
        # 将列表展开为 Sequential 模块
    def forward(self, x):
    # 前向传播函数
        return x + self.conv_block(x)
        # 返回输入与卷积块输出的残差和
class Generator(nn.Module):
# 定义生成器类
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
    # 初始化函数，输入通道、输出通道、残差块数量
        super(Generator, self).__init__()
        # 调用父类初始化
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        # 定义初始层：反射填充、7x7 卷积、实例归一化、ReLU
        in_features = 64
        # 当前特征数为 64
        out_features = in_features*2
        # 下一层特征数为 128
        for _ in range(2):
        # 循环两次进行下采样
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            # 添加步长为 2 的卷积、实例归一化、ReLU
            in_features = out_features
            # 更新输入特征数
            out_features = in_features*2
            # 更新输出特征数
        for _ in range(n_residual_blocks):
        # 添加指定数量的残差块
            model += [ResidualBlock(in_features)]
            # 添加残差块
        out_features = in_features//2
        # 上采样第一层输出特征数减半
        for _ in range(2):
        # 循环两次进行上采样
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            # 添加转置卷积、实例归一化、ReLU
            in_features = out_features
            # 更新输入特征数
            out_features = in_features//2
            # 更新输出特征数
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        # 添加输出层：反射填充、7x7 卷积、Tanh 激活
        self.model = nn.Sequential(*model)
        # 将所有层组合成 Sequential
    def forward(self, x):
    # 前向传播函数
        return self.model(x)
        # 返回生成器输出
class Discriminator(nn.Module):
# 定义判别器类
    def __init__(self, input_nc):
    # 初始化函数，输入通道数
        super(Discriminator, self).__init__()
        # 调用父类初始化
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 第一层：4x4 卷积、LeakyReLU
        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 第二层：卷积、实例归一化、LeakyReLU
        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 第三层：卷积、实例归一化、LeakyReLU
        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 第四层：卷积、实例归一化、LeakyReLU
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        # 输出层：卷积输出单通道
        self.model = nn.Sequential(*model)
        # 组合成 Sequential
    def forward(self, x):
    # 前向传播函数
        x =  self.model(x)
        # 通过模型得到输出
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        # 对空间维度做平均池化并展平