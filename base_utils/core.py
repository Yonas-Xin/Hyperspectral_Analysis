import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
try:
    from osgeo import gdal
except ImportError:
    print('gdal is not used')

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import tempfile
from .gdal_utils import nodata_value,mask_to_vector_gdal,vector_to_mask
gdal.UseExceptions()
class Hyperspectral_Image:
    '''该类的数据形状一律为[H,W,C]'''
    def __init__(self):
        self.dataset, self.rows, self.cols, self.bands = None, None, None, None
        self.filepath_dir = tempfile.gettempdir() # 文件上传与下载的临时目录
        self.no_data = None
        self.sampling_position = None # 二维矩阵，标记了样本的取样点和类别信息
        self.cluster = None
        self.backward_mask = None # [rows, cols]
        self.ori_img = None # [rows, cols, 3]
        self.enhance_data = None # [rows, cols, bands]
        self.enhance_img = None #[rows, cols, 3]

    def __del__(self):
        self.dataset = None # 释放内存
    def init(self, filepath, init_fig=True):
        try:
            dataset = gdal.Open(filepath)
            bands = dataset.RasterCount
            rows, cols = dataset.RasterYSize, dataset.RasterXSize
            self.dataset, self.rows, self.cols, self.bands = dataset, rows, cols, bands
            # self.data = self.get_dataset()
            if init_fig: # 根据需要加载影像数据
                self.init_fig_data()
            return 0 # 代表数据导入成功
        except (AttributeError,RuntimeError):
            return 1

    def create_vector(self,mask,out_file):
        mask_to_vector_gdal(mask, self.dataset.GetGeoTransform(), self.dataset.GetProjection(),
                                   output_shapefile=out_file)
        print(f'shp文件已保存：{out_file}')
    def create_mask(self, input_file):
        return vector_to_mask(input_file, self.dataset.GetGeoTransform(), self.rows, self.cols)

    def init_fig_data(self):
        band = self.dataset.GetRasterBand(1)
        self.no_data = band.GetNoDataValue()
        self.backward_mask = self.ignore_backward()  # 初始化有效像元位置
        self.compose_rgb(r=1, g=2, b=3)

    def update(self,r,g,b,show_enhance_img=False):
        if show_enhance_img:
            self.compose_enhance(r,g,b)
        else:
            self.compose_rgb(r,g,b)

    def compose_rgb(self, r, g, b, stretch=True):
        r_band = self.get_band_data(r)
        g_band = self.get_band_data(g)
        b_band = self.get_band_data(b)
        if stretch:
            r_band = linear_2_percent_stretch(r_band, self.backward_mask)
            g_band = linear_2_percent_stretch(g_band, self.backward_mask)
            b_band = linear_2_percent_stretch(b_band, self.backward_mask)
        rgb = np.dstack([b_band, g_band, r_band]).squeeze().astype(np.float32)
        self.ori_img = np.zeros((self.rows, self.cols, 3)) + 1
        self.ori_img[self.backward_mask] = rgb

    def compose_enhance(self, r, g, b, stretch=True):
        '''这里为了和tif波段组合统一，读取enhance_data波段数据，波段减一'''
        r_band = self.enhance_data[:, :, r-1]
        g_band = self.enhance_data[:, :, g-1]
        b_band = self.enhance_data[:, :, b-1]
        if stretch:
            r_band = linear_2_percent_stretch(r_band, self.backward_mask)
            g_band = linear_2_percent_stretch(g_band, self.backward_mask)
            b_band = linear_2_percent_stretch(b_band, self.backward_mask)
        rgb = np.dstack([b_band, g_band, r_band]).squeeze().astype(np.float32)
        self.enhance_img = np.zeros((self.rows, self.cols, 3)) + 1
        self.enhance_img[self.backward_mask] = rgb

    def get_band_data(self, band_idx):
        """获取指定波段的数据
        :return (rows, cols)"""
        if self.dataset is None:
            return None
        band = self.dataset.GetRasterBand(band_idx)
        return band.ReadAsArray().astype(np.float32)

    def read_pixel(self, row, col):
        pixel_value = []
        for i in range(1,self.bands+1):
            band = self.dataset.GetRasterBand(i)
            value = band.ReadAsArray()[row,col]/10000
            pixel_value.append(value)
        return pixel_value

    def get_dataset(self, scale=1e-4):
        '''返回形状：rows，cols，bands'''
        dataset = self.dataset.ReadAsArray()
        dataset = dataset.transpose(1,2,0).astype(np.float32)*scale
        return dataset

    def ignore_backward(self, nodata_value = nodata_value):
        '''Return:mask[rows*cols,]'''
        dataset = self.get_dataset()
        dataset = np.sum(dataset, axis=-1)
        if self.no_data is not None:
            backward_mask = (dataset == self.no_data*self.bands)
        else:backward_mask = np.zeros((self.rows,self.cols), dtype=bool)
        backward_mask = ~backward_mask
        self.sampling_position = backward_mask
        return backward_mask

    def image_enhance(self, f='PCA', n_components=10, nodata_value=nodata_value):
        dataset = self.get_dataset()
        dataset = dataset[self.backward_mask]
        if f == 'PCA':
            dataset = pca(dataset, n_components=n_components)
        elif f == 'MNF':
            noise = estimate_noise_highpass_non_square(dataset)
            noise = np.cov(noise, rowvar=False) # cov的数据结果只能为float64
            dataset, self.R = mnf_transform_with_steps(dataset, noise, n_components=n_components)
        self.enhance_data = np.full((self.rows, self.cols, n_components), nodata_value, dtype=np.float32)
        self.enhance_data[self.backward_mask] = dataset
        self.compose_enhance(1,2,3)

'''对图像进行拉伸时，要忽略背景值'''
def linear_2_percent_stretch(band_data, mask):
    '''
    线性拉伸
    :param band_data: 单波段数据[rows, cols]
    :param mask: [rows, cols]
    :return: stretched_band[valid_pixels,]
    '''
    band_data = band_data[mask]
    # 计算2%和98%分位数
    lower_percentile = np.percentile(band_data, 2)
    upper_percentile = np.percentile(band_data, 98)
    # 拉伸公式：将数值缩放到 0-1 范围内
    stretched_band = np.clip((band_data - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)
    return stretched_band

def pca(data, n_components=10):
    ''':param data: [rows*cols，bands]'''
    # 计算协方差矩阵
    covariance_matrix = np.cov(data, rowvar=False)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # 按特征值降序排序特征向量
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_idx]
    eigenvectors_selected = -eigenvectors_sorted[:, :n_components] #这里取了负值，实际上正值负值不会影响数据分布，只会影响影像呈现

    data = np.dot(data, eigenvectors_selected)
    return data
def estimate_noise_highpass_non_square(data, sigma=1):
    """
    高通滤波法估计噪声，适用于非方形数据。

    参数:
        data: numpy.ndarray, 输入数据，形状为 (num_samples, bands)
        sigma: float, 高斯滤波器标准差

    返回:
        noise: numpy.ndarray, 噪声数据，形状与输入数据相同
    """
    smoothed = gaussian_filter1d(data, sigma=sigma, axis=1).astype(np.float32)
    noise = data - smoothed
    return noise
def mnf_transform_with_steps(data, noise_cov, n_components=10):
    """
    根据数学推导实现最小噪声分离（MNF）变换。
    参数:
        data: numpy.ndarray, 输入数据，形状为 (rows * cols, bands)
        noise_cov: numpy.ndarray, 噪声数据协方差矩阵

    返回:
        mnf_data: numpy.ndarray, MNF 变换后的数据
    """
    #第一次主成分分析,计算白化数据总协方差矩阵
    eigenvalues, eigenvectors = np.linalg.eigh(noise_cov)  # 特征分解
    order_n = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order_n]
    eigenvectors = -eigenvectors[:, order_n]
    P = eigenvectors @ np.diag(eigenvalues**(-0.5))
    cov = np.cov(data, rowvar=False) # 原始图像协方差矩阵
    data_cov = P.T @ cov @ P # 使用P和原始数据协方差矩阵计算经噪声白化的数据总协方差矩阵, 等同于cov = np.cov(data, rowvar=False)

    # 第二次主成分分析，构造变换矩阵
    eigenvalues, eigenvectors = np.linalg.eigh(data_cov)
    order_w = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order_w]

    #构造变换矩阵
    R = (P @ eigenvectors).astype(np.float32)
    data = (data @ R)[:,:n_components]
    return data, R

def Scaler(data, std = False):
    '''
    对数据进行中心化或者标准化
    std: False-中心化 True-标准化
    '''
    scaler = StandardScaler(with_mean=True, with_std=std)
    return scaler.fit_transform(data)

