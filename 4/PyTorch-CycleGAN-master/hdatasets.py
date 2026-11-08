import glob# 导入文件路径匹配模块
import random# 导入随机数模块
import os# 导入操作系统接口模块
from torch.utils.data import Dataset# 从 PyTorch 导入 Dataset 基类
from PIL import Image# 导入 PIL 图像处理库
import torchvision.transforms as transforms# 导入 torchvision 图像变换模块


class ImageDataset(Dataset):
# 定义图像数据集类，继承 Dataset
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
    # 初始化函数，根目录、变换、是否非配对、模式
        self.transform = transforms.Compose(transforms_)
        # 将变换列表组合成 Compose
        self.unaligned = unaligned
        # 保存是否非配对标志
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        # 获取 A 域所有文件路径并排序
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        # 获取 B 域所有文件路径并排序
    def __getitem__(self, index):
    # 获取指定索引的数据
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        # 打开 A 域图像，索引循环使用
        if image_A.mode != 'RGB':
        # 如果图像不是 RGB 模式
            image_A = image_A.convert('RGB')
            # 转换为 RGB 模式
        item_A = self.transform(image_A)
        # 对 A 图像应用变换
        if self.unaligned:
        # 如果是非配对数据
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            # 随机打开一张 B 域图像
        else:
        # 如果是配对数据
            image_B = Image.open(self.files_B[index % len(self.files_B)])
            # 按索引打开 B 域图像
        if image_B.mode != 'RGB':
        # 如果图像不是 RGB 模式
            image_B = image_B.convert('RGB')
            # 转换为 RGB 模式
        item_B = self.transform(image_B)
        # 对 B 图像应用变换
        return {'A': item_A, 'B': item_B}
        # 返回包含 A 和 B 的字典
    def __len__(self):
    # 返回数据集长度
        return max(len(self.files_A), len(self.files_B))
        # 返回 A 和 B 文件数的最大值