# zlosses.py
# 2026 优化损失函数集合

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
# P0: VGG 结构一致性损失 (DeCGAN, Scientific Reports 2026)
# ============================================================
class ZStructuralConsistencyLoss(nn.Module):
    """
    使用预训练 VGG19 的前若干层提取高层特征，
    计算重建图像与原始图像在特征空间中的 MSE 差异。
    参考：DeCGAN, Scientific Reports, 2026
    """

    def __init__(self, device='cuda', layer_cutoff=12):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:layer_cutoff].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)

    def forward(self, real, reconstructed):
        feat_real = self.vgg(real)
        feat_recon = self.vgg(reconstructed)
        return F.mse_loss(feat_recon, feat_real)


# ============================================================
# P1: 多尺度梯度域约束损失 (CUT-GDC, IEEE Access 2026)
# ============================================================
def z_gradient_domain_loss(fake, real, scales=(1, 2, 4)):
    """
    计算多尺度梯度域约束损失，强制生成图像与真实图像
    在边缘和梯度域上对齐，防止全局布局漂移。
    参考：CUT-GDC, IEEE Access, 2026
    """
    total_loss = 0.0
    for scale in scales:
        if scale > 1:
            fake_s = F.interpolate(fake, scale_factor=1.0 / scale,
                                   mode='bilinear', align_corners=False)
            real_s = F.interpolate(real, scale_factor=1.0 / scale,
                                   mode='bilinear', align_corners=False)
        else:
            fake_s, real_s = fake, real

        fake_dx = fake_s[:, :, :, 1:] - fake_s[:, :, :, :-1]
        real_dx = real_s[:, :, :, 1:] - real_s[:, :, :, :-1]
        fake_dy = fake_s[:, :, 1:, :] - fake_s[:, :, :-1, :]
        real_dy = real_s[:, :, 1:, :] - real_s[:, :, :-1, :]

        loss_scale = F.l1_loss(fake_dx, real_dx) + F.l1_loss(fake_dy, real_dy)
        total_loss = total_loss + loss_scale / scale

    return total_loss / len(scales)


# ============================================================
# P1: 模式多样性损失 (DeCGAN, Scientific Reports 2026)
# ============================================================
class ZModeDiversityLoss(nn.Module):
    """
    通过在输入图像上施加受控扰动，鼓励生成器对不同输入
    产生多样化输出，避免模式崩溃。
    参考：DeCGAN, Scientific Reports, 2026
    """

    def __init__(self, perturbation_scale=0.05):
        super().__init__()
        self.perturbation_scale = perturbation_scale

    def forward(self, generator, input_img):
        noise = torch.randn_like(input_img) * self.perturbation_scale
        perturbed_input = input_img + noise

        output_original = generator(input_img)
        output_perturbed = generator(perturbed_input)

        # 负 L1 距离：距离越大，多样性越高
        diversity = -F.l1_loss(output_perturbed, output_original)
        return diversity


# ============================================================
# P0: 动态自适应权重调度器 (GAN-CLC, npj AI 2026)
# ============================================================
class ZAdaptiveWeightScheduler:
    """
    基于状态反馈控制理论，动态调整损失权重。
    通过监测循环损失与身份损失的变化趋势，
    自适应地扰动闭环系统的平衡点，从动力学角度缓解模式崩溃。
    参考：npj Artificial Intelligence, 2026
    """

    def __init__(self, cycle_init=10.0, identity_init=5.0,
                 cycle_range=(2.0, 20.0), identity_range=(1.0, 15.0),
                 window=20, lr=0.01):
        self.cycle_weight = cycle_init
        self.identity_weight = identity_init
        self.cycle_range = cycle_range
        self.identity_range = identity_range
        self.window = window
        self.lr = lr
        self.cycle_history = []
        self.identity_history = []

    def update(self, loss_cycle, loss_identity):
        self.cycle_history.append(loss_cycle)
        self.identity_history.append(loss_identity)

        if len(self.cycle_history) > self.window:
            self.cycle_history.pop(0)
            self.identity_history.pop(0)

            cycle_trend = self.cycle_history[-1] - self.cycle_history[0]
            identity_trend = self.identity_history[-1] - self.identity_history[0]

            self.cycle_weight *= (1.0 - self.lr * cycle_trend)
            self.cycle_weight = max(self.cycle_range[0],
                                    min(self.cycle_range[1], self.cycle_weight))

            self.identity_weight *= (1.0 - self.lr * identity_trend)
            self.identity_weight = max(self.identity_range[0],
                                       min(self.identity_range[1],
                                           self.identity_weight))

        return self.cycle_weight, self.identity_weight


# ============================================================
# P1: 判别器一致性损失 (HP-GAN, Neural Networks 2026)
# ============================================================
def z_discriminator_consistency_loss(D_A, D_B, real_A, real_B):
    """
    强制两个判别器在评估特征图时保持一致性，
    促进协同学习，增强训练鲁棒性。
    参考：HP-GAN, Neural Networks, 2026
    """
    _, feat_A = D_A(real_A, return_features=True)
    _, feat_B = D_B(real_B, return_features=True)

    feat_A_last = feat_A[-1]
    feat_B_last = feat_B[-1]

    if feat_A_last.shape[2:] != feat_B_last.shape[2:]:
        feat_A_last = F.adaptive_avg_pool2d(feat_A_last,
                                            feat_B_last.shape[2:])

    if feat_A_last.shape[1] != feat_B_last.shape[1]:
        return torch.tensor(0.0, device=feat_A_last.device)

    return F.mse_loss(feat_A_last.mean(dim=[0, 2, 3]),
                      feat_B_last.mean(dim=[0, 2, 3]))


# ============================================================
# 回放缓冲区（保持原版 CycleGAN 设计）
# ============================================================
class ZReplayBuffer:
    """
    保存历史生成的假图，判别器更新时随机使用旧假图，稳定训练。
    """

    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if torch.rand(1).item() > 0.5:
                    i = torch.randint(0, self.max_size - 1, (1,)).item()
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


# ============================================================
# 学习率调度器
# ============================================================
class ZLambdaLR:
    """线性学习率衰减调度器。"""

    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / \
               (self.n_epochs - self.decay_start_epoch)