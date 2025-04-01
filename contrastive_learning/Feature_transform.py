import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
'''保证项目迁移能够正确导包'''

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import affine_grid,grid_sample
import torchvision.transforms.functional as TF
import random
import itertools

def add_gaussian_noise_torch(spectrum, std=0.02):
    """
    使用 PyTorch 给光谱数据添加高斯噪声
    """
    if not isinstance(spectrum, torch.Tensor):
        return ValueError("数据必须是Tensor类型")
    device = spectrum.device
    noise = torch.randn_like(spectrum, device=device) * std
    return spectrum + noise

class BatchAugment(nn.Module):
    '''使用时最好将图像转为float类型'''
    def __init__(self,
                 flip_prob: float = 0.5,
                 rotate_prob: float = 0.5,
                 brightness_range: tuple = (0.75, 1.25),
                 contrast_range: tuple = (0.75, 1.25),
                 # saturation_range: tuple = (0.8, 1.2), # 饱和度
                 hue_range: tuple = (-0.2, 0.2)):
        super().__init__()
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        # self.saturation_range = saturation_range
        self.hue_range = hue_range

    def augment_batch(self, x: Tensor) -> Tensor:
        """
        批量数据增强, 避免循环处理
        输入形状: (batch, C, H, W)
        输出形状: (batch, C, H, W)
        """
        device = x.device
        batch_size = x.shape[0]

        # 随机翻转（批量处理）
        if self.flip_prob > 0:
            flip_mask = torch.rand(batch_size, device=device) < self.flip_prob
            direction = torch.rand(batch_size, device=device) < 0.5

            # 水平翻转
            h_flip_mask = flip_mask & direction
            if h_flip_mask.any():
                x[h_flip_mask] = TF.hflip(x[h_flip_mask])

            # 垂直翻转
            v_flip_mask = flip_mask & ~direction
            if v_flip_mask.any():
                x[v_flip_mask] = TF.vflip(x[v_flip_mask])

        # 随机旋转（批量处理）
        if self.rotate_prob > 0:
            rotate_mask = torch.rand(batch_size, device=device) < self.rotate_prob
            angles = torch.zeros(batch_size, device=device)
            angle_choices = torch.tensor(
                [0, 30, 60, 90, 120, 180, 210, 240, 270, 300, 330],
                device=device,
                dtype=torch.float
            )
            selected = torch.randint(0, 11, (batch_size,), device=device)
            angles[rotate_mask] = angle_choices[selected[rotate_mask]]

            # 构造旋转矩阵
            theta = torch.deg2rad(angles)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            rotation_matrix = torch.zeros((batch_size, 2, 3), device=device)
            rotation_matrix[:, 0, 0] = cos_theta
            rotation_matrix[:, 0, 1] = -sin_theta
            rotation_matrix[:, 1, 0] = sin_theta
            rotation_matrix[:, 1, 1] = cos_theta

            # 应用批量旋转
            grid = affine_grid(rotation_matrix, x.size(), align_corners=False)
            x_rotated = grid_sample(x, grid, align_corners=False)
            x = torch.where(rotate_mask[:, None, None, None], x_rotated, x)

        # 亮度调整（向量化）
        if self.brightness_range != (1.0, 1.0):
            brightness_factors = torch.empty(batch_size, device=device).uniform_(*self.brightness_range)
            x = x * brightness_factors.view(-1, 1, 1, 1)
            x = torch.clamp(x, 0, 1)

        # 对比度调整（向量化）
        if self.contrast_range != (1.0, 1.0):
            contrast_factors = torch.empty(batch_size, device=device).uniform_(*self.contrast_range)
            mean = torch.mean(x, dim=(2, 3), keepdim=True)
            x = (x - mean) * contrast_factors.view(-1, 1, 1, 1) + mean
            x = torch.clamp(x, 0, 1)

        # 色调调整（逐图像处理，但在 GPU 上）
        if self.hue_range != (0.0, 0.0):
            hue_factors = torch.empty(batch_size, device=device).uniform_(*self.hue_range)
            for i in range(batch_size):
                x[i] = TF.adjust_hue(x[i], hue_factors[i].item())
        return x
    def forward(self, x: Tensor) -> Tensor:
        return self.augment_batch(x.clone().float()) # 克隆数据防止原始数据被改变

class BatchAugment_3d(nn.Module):
    '''使用时最好将图像转为float类型'''
    def __init__(self,
                 flip_prob: float = 0.5,
                 rot_prob: float = 0.5,
                 gaussian_noise_std: tuple = (0.006, 0.012)):
        super().__init__()
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.gaussian_noise_std = gaussian_noise_std

    def rot_90(self,x,k=1):
        return torch.rot90(x, dims=(3,4), k=k)
    def flip(self,x,dims=[3]):
        return torch.flip(x, dims=dims)
    def augment_batch(self, x: Tensor) -> Tensor:
        """
        批量数据增强, 避免循环处理
        输入形状: (batch, 1, H, W, C)
        输出形状: (batch, 1, H, W, C)
        """
        device = x.device
        batch_size,_,C,H,W = x.shape
        # 随机翻转（批量处理）
        if self.flip_prob > 0:
            h_flip_mask = torch.rand(batch_size, device=device) < self.flip_prob
            v_flip_mask = (torch.rand(batch_size, device=device) < self.flip_prob) & (~h_flip_mask)
            rot_90_mask = (~h_flip_mask & ~v_flip_mask) | (torch.rand(batch_size, device=device) < self.rot_prob)
            rot_270_mask = (torch.rand(batch_size, device=device) < self.rot_prob) & (~rot_90_mask)
            rot_180_mask = (torch.rand(batch_size, device=device) < self.rot_prob) & (~rot_90_mask) & (~rot_270_mask)
            if h_flip_mask.any():
                x[h_flip_mask] = self.flip(x[h_flip_mask], dims=[3])
            if v_flip_mask.any():
                x[v_flip_mask] = self.flip(x[v_flip_mask], dims=[4])

            if rot_90_mask.any():
                x[rot_90_mask] = self.rot_90(x[rot_90_mask], k=1)
            if rot_270_mask.any():
                x[rot_270_mask] = self.rot_90(x[rot_270_mask], k=3)
            if rot_180_mask.any():
                x[rot_180_mask] = self.rot_90(x[rot_180_mask], k=2)
        std = random.uniform(self.gaussian_noise_std[0], self.gaussian_noise_std[1]) # 随机选择噪声强度
        return add_gaussian_noise_torch(x, std=std)
    def forward(self, x: Tensor) -> Tensor:
        return self.augment_batch(x.clone().float())

    def generate_enhance_list(self, factor=10):
        flip_options = [[2], [3], None]
        rot_options = [1, 2, 3, None]
        gaussian_options = [0.005, 0.010, 0.015, None]
        all_combinations = list(itertools.product(flip_options, rot_options, gaussian_options))[:-1]
        self.enhance_order = random.sample(all_combinations, factor)


    def order(self, x: Tensor, idx=0): # 用来对数据进行指定形式的增强，扩展数据集。
        flip, rot, noise_std = self.enhance_order[idx]
        if flip is not None:
            x = self.flip(x, dims=flip)
        if rot is not None:
            x = self.rot_90(x, k=rot)
        if noise_std is not None:
            x = add_gaussian_noise_torch(x, std=noise_std)
        return x


if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader
    from base_utils import Hyperspectral_Image, gdal_utils
    import torch
    from tqdm import tqdm
    from cnn_model import Block_Generator,show_img
    batch = 10
    img = Hyperspectral_Image()
    img.init("D:\Data\yanjiuqu\预处理\crop_for_research.dat", init_fig=False)
    dataset = Block_Generator(img.get_dataset(scale=1e-4).transpose(2,0,1), block_size=25)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True)
    augment = BatchAugment_3d()
    idx = 0
    for block in tqdm(dataloader, total=len(dataloader)):
        block1 = augment(block)
        show_img(data = block[0, 0, :, :, :])
        show_img(data = block1[0, 0, :, :, :])
        print(0)
