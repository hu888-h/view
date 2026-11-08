# view — 图像处理课程实验

六次实验的代码与报告。核心实现是 CycleGAN（第 4 次）与超分复原（第 5 次），
其余为图像处理基础、分类、目标检测。

## 实验清单

| 目录 | 主题 | 内容 |
|---|---|---|
| `1/` | 图像处理基础 | 滤波/卷积实现 `temp-filter.py`，对比 OpenCV 内置实现 |
| `2/` | 图像分类 | 报告（实验文档）+ 分类流程说明 |
| `3/` | （压缩包） | 已排除，见下方"未入库内容" |
| `4/` | **生成对抗网络** | CycleGAN 完整实现，马→斑马非成对转换 |
| `5/` | **图像复原** | 超分辨率（SRCNN/SRGAN 系），含训练/测试/推理脚本 |
| `6/` | 图像检测 | 报告（实验文档） |

## 重点：`4/PyTorch-CycleGAN-master`

```
models.py / zmodels.py     生成器与判别器（ResNet 结构生成器 + PatchGAN 判别器）
datasets.py / hdatasets.py 非成对数据加载（horse2zebra），含翻转/裁剪增强
train.py / ztrain.py       训练循环：对抗损失 + 循环一致性损失 + identity 损失
test.py / htest.py         推理与拼图输出
utils.py                   ReplayBuffer、权重初始化、图像池
zconfig.py                 超参与路径集中配置
```

自写版本（`z*.py`）在原始实现基础上做的改动：损失项与网络深度做了参数化，
方便在显存受限时缩小模型跑通流程。

### 训练

```bash
cd 4/PyTorch-CycleGAN-master
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan
```

权重（`pretrained_netG_A2B.pth` 等，单文件 45MB）与 `datasets/`（118MB 图片）
未入库，需要时从原仓库或自行训练获取。

## 重点：`5/` 图像复原

```
models.py   网络定义
datasets.py 数据集封装（91-image / Set5 的 x4 上下采样对）
train.py    训练
test.py     推理 + PSNR/SSIM 评估
data/       少量示例图（ppt3.bmp, zebra.bmp, butterfly_GT.bmp）
```

`dataset/91-image_x4.h5`（190MB）与 `Set5_x4.h5` 是打包好的训练数据，未入库。
`datasets.py` 里 HDF5 与目录两种加载方式都保留了，把 h5 放回 `dataset/` 即可跑。

## 未入库内容（体积原因）

| 路径 | 大小 | 说明 |
|---|---|---|
| `4/PyTorch-CycleGAN-master.zip` | 212MB | 源码压缩包，源码已解压入库 |
| `4/4.zip` | 212MB | 同上 |
| `3/压缩文件.zip` | 128MB | 实验三素材 |
| `4/*/datasets/horse2zebra/` | 118MB | 训练数据集（2661 张图） |
| `5/dataset/*.h5` | 185MB | 打包数据集 |
| `4/*/models/*.pth` | 112MB | 预训练权重 |

## 环境

```bash
pip install torch torchvision h5py numpy pillow scikit-image
```

## 许可

课业代码，MIT。`PyTorch-CycleGAN-master` 源自公开实现，版权归原作者。
