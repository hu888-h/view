# zconfig.py
# 2026 CycleGAN 优化配置中心

class ZConfig:
    """
    统一配置管理类。
    所有可调参数集中在此，方便训练脚本按需引用或覆盖。
    """

    # ==================== 基础训练参数 ====================
    epoch = 0                          # 起始 epoch
    n_epochs = 40                      # 总训练轮数
    batchSize = 1                      # 批次大小
    dataroot = './datasets/horse2zebra/'  # 数据集根目录
    lr = 0.0002                        # 初始学习率
    decay_epoch = 10                   # 学习率开始线性衰减的 epoch
    size = 256                         # 训练图像尺寸（正方形）
    input_nc = 3                       # 输入通道数
    output_nc = 3                      # 输出通道数
    n_cpu = 4                          # DataLoader 线程数
    cuda = True                        # 是否启用 GPU

    # ==================== 优化模块开关 ====================
    # ---------- P0 级别（低成本、高收益） ----------
    # 谱归一化：稳定判别器训练 (DeCGAN, Scientific Reports 2026)
    use_spectral_norm = True

    # VGG 结构一致性损失 (DeCGAN, Scientific Reports 2026)
    use_structural_loss = True
    lambda_struct = 0.1                # 结构损失权重

    # 动态自适应损失权重 (GAN-CLC, npj AI 2026)
    use_adaptive_weight = True
    cycle_weight_init = 10.0           # 循环损失初始权重
    identity_weight_init = 5.0         # 身份损失初始权重

    # ---------- P1 级别（中等改动、收益明显） ----------
    # 多尺度梯度域约束 (CUT-GDC, IEEE Access 2026)
    use_gradient_loss = True
    lambda_gradient = 2.0              # 梯度损失权重

    # 判别器一致性 (HP-GAN, Neural Networks 2026)
    use_disc_consistency = True
    lambda_consistency = 0.1           # 一致性损失权重

    # 模式多样性损失 (DeCGAN, Scientific Reports 2026)
    use_mode_diversity = True
    lambda_diversity = 0.5             # 多样性损失权重

    # ---------- P2 级别（架构级改动） ----------
    # 均衡跳跃连接残差块 (CCGAN-BSNRL, Royal Society OS 2026)
    use_balanced_skip = True

    # 轻量级 ViT 块 (MViT-StrokeGAN, 2026)
    use_vit = False                    # 默认关闭，开启需更多显存
    vit_at_block = 4                   # 在第几个残差块后插入 ViT

    # ==================== 训练辅助参数 ====================
    # 自适应权重调度窗口
    adaptive_window = 20
    adaptive_lr = 0.01
    cycle_weight_range = (2.0, 20.0)
    identity_weight_range = (1.0, 15.0)

    # VGG 感知损失使用的层数
    vgg_layer_cutoff = 12

    # 模式多样性扰动幅度
    diversity_perturbation = 0.05

    # 模型保存目录
    model_save_dir = './models'

    # 日志打印间隔（每多少个 batch 打印一次）
    log_interval = 50