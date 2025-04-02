import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
'''保证项目迁移能够正确导包'''

from torch.utils.data import Dataset
from base_utils.Dataloader_X import DataLoaderX
from base_utils import gdal_utils
from base_utils.core import Hyperspectral_Image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
class IS_Generator(Dataset):
    '''构造用于1d+2d编码器的输入'''
    def __init__(self, whole_space, whole_spectral, block_size=25):
        '''将整幅图像裁剪成为适用于模型输入的数据集形式
        whole_space[H,W,3]'''
        _, self.rows, self.cols = whole_space.shape
        self.block_size = block_size
        if block_size % 2 == 0:
            left_top = int(block_size / 2 - 1)
            right_bottom = int(block_size / 2)
        else:
            left_top = int(block_size // 2)
            right_bottom = int(block_size // 2)
        self.whole_space = np.pad(whole_space, [(left_top, right_bottom), (left_top, right_bottom), (0, 0)], 'constant')
        self.whole_spectral = whole_spectral

    def __len__(self):
        return self.rows*self.cols
    def __getitem__(self, idx):
        """
        根据索引返回图像及其光谱
        """
        row = idx//self.cols
        col = idx%self.cols # 根据索引生成二维索引
        block, spectral = self.get_samples(row, col)

        # 转换为 PyTorch 张量
        block = torch.from_numpy(block).float()
        spectral = torch.from_numpy(spectral).float()
        return block, spectral

    def get_samples(self,row,col):
        block = self.whole_space[:, row:row + self.block_size, col:col + self.block_size]
        spectral = self.whole_spectral[:, row, col:col+1]
        return block,spectral

class Block_Generator(Dataset):
    '''构造用于3D编码器的输入'''
    def __init__(self, data, block_size=25):
        '''将整幅图像裁剪成为适用于模型输入的数据集形式
        data[C,H,W]'''
        _, self.rows, self.cols = data.shape
        self.block_size = block_size
        if block_size % 2 == 0:
            left_top = int(block_size / 2 - 1)
            right_bottom = int(block_size / 2)
        else:
            left_top = int(block_size // 2)
            right_bottom = int(block_size // 2)
        self.whole_space = np.pad(data, [(0, 0),(left_top, right_bottom), (left_top, right_bottom)], 'constant')

    def __len__(self):
        return self.rows*self.cols
    def __getitem__(self, idx):
        """
        根据索引返回图像及其光谱
        """
        row = idx//self.cols
        col = idx%self.cols # 根据索引生成二维索引
        block = self.get_samples(row, col)
        # 转换为 PyTorch 张量
        block = torch.from_numpy(block).float()
        return block

    def get_samples(self,row,col):
        block = self.whole_space[:,row:row + self.block_size, col:col + self.block_size]
        return block
def show_img(data:torch.Tensor):
    data = data.numpy()
    image = data[(29,19,9),:,:].transpose(1,2,0)
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    from Models import CNN_3d
    batch = 4
    model_path  = '.models/'
    img = Hyperspectral_Image()
    img.init(r"D:\Programing\pythonProject\data_store\research_area1.dat", init_fig=False)
    dataset = Block_Generator(img.get_dataset(scale=1e-4).transpose(2,0,1), block_size=25)
    dataloader = DataLoaderX(dataset, batch_size=batch, shuffle=False, pin_memory=True, num_workers=4)

    model = CNN_3d(out_embedding=20, out_classes=8)
    # dic = torch.load(model_path, weights_only=True)
    # model.load_state_dict(dic)
    device = torch.device('cuda')
    model.to(device)
    predict_map = np.empty((img.rows*img.cols,), dtype=np.int16)
    idx = 0

    model.eval()
    with torch.no_grad():
        for block in tqdm(dataloader, total=len(dataloader)):
            batch = block.shape[0]
            show_img(block[0,:,:,:] .cpu())# 展示滑动窗口
            model.eval()
            block = block.unqueeze(1).to(device)
            outputs = model(block)
            _, predicted = torch.max(outputs, 1)
            predict_map[idx:idx+batch,] = predicted.cpu().numpy()
            idx += batch
    predict_map = predict_map.reshape(img.rows, img.cols)
    np.savez_compressed('predict_map.npz',data=predict_map)
    map = gdal_utils.label_to_rgb(predict_map)
    plt.imshow(map)
    plt.axis('off')
    plt.show()