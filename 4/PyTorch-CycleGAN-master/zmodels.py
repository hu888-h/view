# zmodels.py
# 2026 优化版生成器与判别器

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ============================================================
# P2: 均衡跳跃连接残差块 (CCGAN-BSNRL, Royal Society OS 2026)
# ============================================================
class BalancedSkipResidualBlock(nn.Module):
    """
    均衡跳跃连接残差块。
    核心思想：对多路跳跃连接进行加权平衡，权重因子为 1/w
    （w 为跳跃连接数量），使残差分布更加均匀，有助于梯度流动和结构保持。
    参考：Salehin et al., Royal Society Open Science, 2026
    """

    def __init__(self, in_features, n_skips=2):
        super().__init__()
        self.n_skips = n_skips

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

        # 均衡权重：每个跳跃连接贡献 1/w
        self.skip_weight = 1.0 / n_skips

    def forward(self, x):
        out = self.conv_block(x)
        # 均衡跳跃连接：原始输入 + 卷积输出，各占 1/w 权重
        return x * self.skip_weight + out * self.skip_weight + \
               x * (1 - 2 * self.skip_weight)


# ============================================================
# 标准残差块（备用）
# ============================================================
class StandardResidualBlock(nn.Module):
    """标准 CycleGAN 残差块。"""

    def __init__(self, in_features):
        super().__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# ============================================================
# P2: 轻量级 Transformer 注意力块 (MViT-StrokeGAN, 2026)
# ============================================================
class LightweightViTBlock(nn.Module):
    """
    轻量级 ViT 块，用于在生成器中嵌入全局注意力，
    增强上下文理解能力，同时保持计算效率。
    参考：MViT-StrokeGAN, 2026
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)          # B, HW, C
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + attn_out)
        x_flat = self.norm2(x_flat + self.ffn(x_flat))
        return x_flat.transpose(1, 2).view(B, C, H, W)


# ============================================================
# 2026 生成器（集成均衡跳跃连接 + 可选 ViT）
# ============================================================
class ZGenerator(nn.Module):
    """
    2026 优化版生成器。
    - 默认使用均衡跳跃连接残差块
    - 可选在指定残差块后插入轻量级 ViT 块
    """

    def __init__(self, input_nc, output_nc, n_residual_blocks=9,
                 use_balanced_skip=True, use_vit=False, vit_at_block=4):
        super().__init__()

        # ---------- 初始下采样前端 ----------
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # ---------- 两次下采样 ----------
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # ---------- 残差块（可选 ViT） ----------
        for i in range(n_residual_blocks):
            if use_balanced_skip:
                model += [BalancedSkipResidualBlock(in_features)]
            else:
                model += [StandardResidualBlock(in_features)]

            if use_vit and i == vit_at_block:
                model += [LightweightViTBlock(in_features)]

        # ---------- 两次上采样 ----------
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # ---------- 输出层 ----------
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ============================================================
# P0: 谱归一化判别器 (DeCGAN, Scientific Reports 2026)
# ============================================================
class ZDiscriminator(nn.Module):
    """
    2026 优化版判别器。
    在判别器的所有卷积层上应用谱归一化，
    将权重矩阵除以其谱范数，强制执行 Lipschitz 连续性约束，
    稳定判别器训练，防止对输入微小变化过度敏感。
    参考：DeCGAN, Scientific Reports, 2026
    """

    def __init__(self, input_nc, use_spectral_norm=True):
        super().__init__()

        def conv_layer(in_ch, out_ch, k, s=1, p=0):
            layer = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p)
            if use_spectral_norm:
                return spectral_norm(layer)
            return layer

        model = [
            conv_layer(input_nc, 64, 4, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            conv_layer(64, 128, 4, s=2, p=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            conv_layer(128, 256, 4, s=2, p=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            conv_layer(256, 512, 4, s=1, p=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [conv_layer(512, 1, 4, s=1, p=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x, return_features=False):
        """
        return_features=True 时返回 (判别输出, 中间特征列表)，
        用于判别器一致性损失计算。
        """
        if return_features:
            features = []
            for i, layer in enumerate(self.model):
                x = layer(x)
                if isinstance(layer, nn.LeakyReLU) and i < len(self.model) - 2:
                    features.append(x)
            pooled = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
            return pooled, features

        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# ============================================================
# 权重初始化函数
# ============================================================
def z_weights_init_normal(m):
    """正态分布权重初始化，适用于 Conv 与 BatchNorm 层。"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)