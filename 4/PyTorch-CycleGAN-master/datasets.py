import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]
        image_A = Image.open(path_A)
        if image_A.mode != 'RGB':
            image_A = image_A.convert('RGB')
        item_A = self.transform(image_A)


        if self.unaligned:
            path_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            path_B = self.files_B[index % len(self.files_B)]
        image_B = Image.open(path_B)

        if image_B.mode != 'RGB':
            image_B = image_B.convert('RGB')
        item_B = self.transform(image_B)

        return {
            'A': item_A,
            'B': item_B,
            'A_path': os.path.basename(path_A),
            'B_path': os.path.basename(path_B),
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))