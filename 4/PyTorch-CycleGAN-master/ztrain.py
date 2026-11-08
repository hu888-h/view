# ztrain.py
# 2026 优化版 CycleGAN 训练脚本

import argparse
import itertools
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ---------- 引入本地模块 ----------
from zconfig import ZConfig
from zmodels import (
    ZGenerator, ZDiscriminator, z_weights_init_normal
)
from zlosses import (
    ZStructuralConsistencyLoss, z_gradient_domain_loss,
    ZModeDiversityLoss, ZAdaptiveWeightScheduler,
    z_discriminator_consistency_loss, ZReplayBuffer, ZLambdaLR
)
from datasets import ImageDataset


# ============================================================
# 命令行参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # 基础训练参数
    parser.add_argument('--epoch', type=int, default=ZConfig.epoch)
    parser.add_argument('--n_epochs', type=int, default=ZConfig.n_epochs)
    parser.add_argument('--batchSize', type=int, default=ZConfig.batchSize)
    parser.add_argument('--dataroot', type=str, default=ZConfig.dataroot)
    parser.add_argument('--lr', type=float, default=ZConfig.lr)
    parser.add_argument('--decay_epoch', type=int, default=ZConfig.decay_epoch)
    parser.add_argument('--size', type=int, default=ZConfig.size)
    parser.add_argument('--input_nc', type=int, default=ZConfig.input_nc)
    parser.add_argument('--output_nc', type=int, default=ZConfig.output_nc)
    parser.add_argument('--cuda', action='store_true', default=ZConfig.cuda)
    parser.add_argument('--n_cpu', type=int, default=ZConfig.n_cpu)

    # 优化模块开关
    parser.add_argument('--use_spectral_norm', type=bool,
                        default=ZConfig.use_spectral_norm)
    parser.add_argument('--use_structural_loss', type=bool,
                        default=ZConfig.use_structural_loss)
    parser.add_argument('--use_gradient_loss', type=bool,
                        default=ZConfig.use_gradient_loss)
    parser.add_argument('--use_adaptive_weight', type=bool,
                        default=ZConfig.use_adaptive_weight)
    parser.add_argument('--use_disc_consistency', type=bool,
                        default=ZConfig.use_disc_consistency)
    parser.add_argument('--use_mode_diversity', type=bool,
                        default=ZConfig.use_mode_diversity)
    parser.add_argument('--use_balanced_skip', type=bool,
                        default=ZConfig.use_balanced_skip)
    parser.add_argument('--use_vit', type=bool, default=ZConfig.use_vit)

    return parser.parse_args()


# ============================================================
# 主训练流程
# ============================================================
def main():
    opt = parse_args()

    # ---------- 设备选择 ----------
    opt.cuda = opt.cuda or torch.cuda.is_available()
    device = torch.device('cuda' if opt.cuda else 'cpu')
    print(f"Using {'GPU' if opt.cuda else 'CPU'} computation.")
    print(f"Options: {vars(opt)}")

    # ========================================================
    # 模型初始化
    # ========================================================
    netG_A2B = ZGenerator(
        opt.input_nc, opt.output_nc,
        use_balanced_skip=opt.use_balanced_skip,
        use_vit=opt.use_vit,
        vit_at_block=ZConfig.vit_at_block
    )
    netG_B2A = ZGenerator(
        opt.output_nc, opt.input_nc,
        use_balanced_skip=opt.use_balanced_skip,
        use_vit=opt.use_vit,
        vit_at_block=ZConfig.vit_at_block
    )
    netD_A = ZDiscriminator(
        opt.input_nc,
        use_spectral_norm=opt.use_spectral_norm
    )
    netD_B = ZDiscriminator(
        opt.output_nc,
        use_spectral_norm=opt.use_spectral_norm
    )

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # 权重初始化（跳过谱归一化包裹层，由原层初始化即可）
    netG_A2B.apply(z_weights_init_normal)
    netG_B2A.apply(z_weights_init_normal)
    netD_A.apply(z_weights_init_normal)
    netD_B.apply(z_weights_init_normal)

    # ========================================================
    # 损失函数
    # ========================================================
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # P0: VGG 结构一致性损失
    criterion_struct = None
    if opt.use_structural_loss:
        criterion_struct = ZStructuralConsistencyLoss(
            device=str(device),
            layer_cutoff=ZConfig.vgg_layer_cutoff
        )

    # P1: 模式多样性损失
    criterion_diversity = None
    if opt.use_mode_diversity:
        criterion_diversity = ZModeDiversityLoss(
            perturbation_scale=ZConfig.diversity_perturbation
        )

    # P0: 自适应权重调度器
    weight_scheduler = None
    if opt.use_adaptive_weight:
        weight_scheduler = ZAdaptiveWeightScheduler(
            cycle_init=ZConfig.cycle_weight_init,
            identity_init=ZConfig.identity_weight_init,
            cycle_range=ZConfig.cycle_weight_range,
            identity_range=ZConfig.identity_weight_range,
            window=ZConfig.adaptive_window,
            lr=ZConfig.adaptive_lr
        )

    # ========================================================
    # 优化器与学习率调度
    # ========================================================
    optimizer_G = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=opt.lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999)
    )
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999)
    )

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=ZLambdaLR(opt.n_epochs, opt.epoch,
                            opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A,
        lr_lambda=ZLambdaLR(opt.n_epochs, opt.epoch,
                            opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B,
        lr_lambda=ZLambdaLR(opt.n_epochs, opt.epoch,
                            opt.decay_epoch).step
    )

    # ========================================================
    # 回放缓冲区
    # ========================================================
    fake_A_buffer = ZReplayBuffer()
    fake_B_buffer = ZReplayBuffer()

    # ========================================================
    # 数据加载
    # ========================================================
    transforms_ = [
        transforms.Resize(int(opt.size * 1.12),
                          transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    dataloader = DataLoader(
        ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu
    )

    # ========================================================
    # 训练循环
    # ========================================================
    os.makedirs(ZConfig.model_save_dir, exist_ok=True)

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # ------------------------------------------------
            # 生成器更新
            # ------------------------------------------------
            optimizer_G.zero_grad()

            # --- 身份损失 ---
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)

            # --- 对抗损失 ---
            fake_B = netG_A2B(real_A)
            pred_fake_B = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(
                pred_fake_B, torch.ones_like(pred_fake_B)
            )

            fake_A = netG_B2A(real_B)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(
                pred_fake_A, torch.ones_like(pred_fake_A)
            )

            # --- 循环一致性损失 ---
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

            # --- P0: VGG 结构一致性损失 ---
            loss_struct = torch.tensor(0.0, device=device)
            if criterion_struct is not None:
                loss_struct = (
                    criterion_struct(real_A, recovered_A) +
                    criterion_struct(real_B, recovered_B)
                )

            # --- P1: 梯度域约束损失 ---
            loss_gradient = torch.tensor(0.0, device=device)
            if opt.use_gradient_loss:
                loss_gradient = (
                    z_gradient_domain_loss(fake_B, real_B) +
                    z_gradient_domain_loss(fake_A, real_A)
                )

            # --- P1: 模式多样性损失 ---
            loss_diversity = torch.tensor(0.0, device=device)
            if criterion_diversity is not None:
                loss_diversity = (
                    criterion_diversity(netG_A2B, real_A) +
                    criterion_diversity(netG_B2A, real_B)
                )

            # --- P0: 动态自适应权重 ---
            if weight_scheduler is not None:
                cycle_w, identity_w = weight_scheduler.update(
                    (loss_cycle_ABA + loss_cycle_BAB).item(),
                    (loss_identity_A + loss_identity_B).item()
                )
            else:
                cycle_w = ZConfig.cycle_weight_init
                identity_w = ZConfig.identity_weight_init

            # --- 生成器总损失 ---
            loss_G = (
                identity_w * (loss_identity_A + loss_identity_B) +
                loss_GAN_A2B + loss_GAN_B2A +
                cycle_w * (loss_cycle_ABA + loss_cycle_BAB) +
                ZConfig.lambda_struct * loss_struct +
                ZConfig.lambda_gradient * loss_gradient +
                ZConfig.lambda_diversity * loss_diversity
            )
            loss_G.backward()
            optimizer_G.step()

            # ------------------------------------------------
            # 判别器 A 更新
            # ------------------------------------------------
            optimizer_D_A.zero_grad()

            pred_real_A = netD_A(real_A)
            loss_D_real_A = criterion_GAN(
                pred_real_A, torch.ones_like(pred_real_A)
            )

            fake_A_buf = fake_A_buffer.push_and_pop(fake_A)
            pred_fake_A = netD_A(fake_A_buf.detach())
            loss_D_fake_A = criterion_GAN(
                pred_fake_A, torch.zeros_like(pred_fake_A)
            )

            # --- P1: 判别器一致性损失 ---
            loss_consistency = torch.tensor(0.0, device=device)
            if opt.use_disc_consistency:
                loss_consistency = z_discriminator_consistency_loss(
                    netD_A, netD_B, real_A, real_B
                )

            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5 + \
                       ZConfig.lambda_consistency * loss_consistency
            loss_D_A.backward()
            optimizer_D_A.step()

            # ------------------------------------------------
            # 判别器 B 更新
            # ------------------------------------------------
            optimizer_D_B.zero_grad()

            pred_real_B = netD_B(real_B)
            loss_D_real_B = criterion_GAN(
                pred_real_B, torch.ones_like(pred_real_B)
            )

            fake_B_buf = fake_B_buffer.push_and_pop(fake_B)
            pred_fake_B = netD_B(fake_B_buf.detach())
            loss_D_fake_B = criterion_GAN(
                pred_fake_B, torch.zeros_like(pred_fake_B)
            )

            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # ------------------------------------------------
            # 日志打印
            # ------------------------------------------------
            if (i + 1) % ZConfig.log_interval == 0:
                print(
                    f"[Epoch {epoch}/{opt.n_epochs}] "
                    f"[Batch {i+1}/{len(dataloader)}] "
                    f"[D_A: {loss_D_A.item():.4f}] "
                    f"[D_B: {loss_D_B.item():.4f}] "
                    f"[G: {loss_G.item():.4f}] "
                    f"[cycle_w: {cycle_w:.2f}] "
                    f"[id_w: {identity_w:.2f}]"
                )

        # ------------------------------------------------
        # epoch 结束：更新学习率
        # ------------------------------------------------
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # ------------------------------------------------
        # epoch 结束：保存模型
        # ------------------------------------------------
        torch.save(netG_A2B.state_dict(),
                   os.path.join(ZConfig.model_save_dir, 'znetG_A2B.pth'))
        torch.save(netG_B2A.state_dict(),
                   os.path.join(ZConfig.model_save_dir, 'znetG_B2A.pth'))
        torch.save(netD_A.state_dict(),
                   os.path.join(ZConfig.model_save_dir, 'znetD_A.pth'))
        torch.save(netD_B.state_dict(),
                   os.path.join(ZConfig.model_save_dir, 'znetD_B.pth'))
        print(f"[Epoch {epoch}] Models saved to {ZConfig.model_save_dir}")

    print("Training complete.")


if __name__ == '__main__':
    main()