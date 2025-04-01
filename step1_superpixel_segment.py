import os.path
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from base_utils.core import *
from skimage.segmentation import slic
from base_utils.gdal_utils import mask_to_vector_gdal
import matplotlib.pyplot as plt

def superpixel_segmentation(data_pca, n_segments=1024, compactness=10):
    # img_lab = color.rgb2lab(data_pca) # rgb转lab
    segments = slic(
        data_pca,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
    )
    return segments

def apply_superpixel_mean_color(image, segments):
    '''计算超像素颜色均值，用于绘制超像素图'''
    output = np.zeros_like(image)
    for seg_val in np.unique(segments):  # 遍历所有超像素区域
        mask = segments == seg_val  # 选出当前超像素区域
        mean_color = image[mask].mean(axis=0)  # 计算均值颜色
        output[mask] = mean_color  # 赋值给对应的区域
    return output

def plot_superpixel_figure(ori_img, segments):
    '''
    绘制超像素分割图
    :param ori_img: （rows,cols,3）原始影像
    :param segments: (rows,cols)分割标签
    :return:
    '''
    rgb = apply_superpixel_mean_color(ori_img,segments)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
    print()

def calculate_spectral_heterogeneity(dataset, segments):
    """
    计算每个超像素区域的光谱异质性（基于方差）

    参数：
        dataset: 高光谱数据 (rows, cols, bands)
        segments: 超像素标签图 (rows, cols)

    返回：
        hetero_dict: {超像素ID: 异质性分数}
    """
    unique_segments = np.unique(segments)
    hetero_dict = {}
    for seg_id in unique_segments:
        # 获取当前超像素的像素坐标
        mask = (segments == seg_id)
        region_data = dataset[mask]  # (n_pixels, bands)
        # 计算光谱方差（各波段方差均值）
        if region_data.shape[0] == 0:
            hetero = 0.0
        else:
            band_variances = np.var(region_data, axis=0)
            hetero = np.mean(band_variances)
        hetero_dict[seg_id] = hetero
    return hetero_dict

def generate_adaptive_mask(segments, hetero_dict, base_samples_ratio=0.005, max_samples_ratio=0.005):
    """
    生成自适应采样Mask
    参数：
        dataset: 高光谱数据 (rows, cols, bands)
        segments: 超像素标签图 (rows, cols)
        hetero_dict: 异质性字典
        base_samples: 基础采样数
        max_samples: 最大采样数

    返回：
        mask: 采样点标记矩阵 (rows, cols)
    """
    rows, cols = segments.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    unique_segments = np.unique(segments)
    # 归一化异质性分数到[0,1]
    hetero_values = np.array(list(hetero_dict.values()))
    hetero_norm = (hetero_values - np.min(hetero_values)) / (np.max(hetero_values) - np.min(hetero_values) + 1e-6)
    hetero_dict_norm = {seg_id: hetero_norm[i] for i, seg_id in enumerate(unique_segments)}
    for seg_id in unique_segments:
        # 获取当前超像素区域坐标
        y, x = np.where(segments == seg_id)
        coordinates = list(zip(y, x))
        if len(coordinates) == 0:
            continue
        base_samples = base_samples_ratio * len(coordinates)
        max_samples = max_samples_ratio * len(coordinates)
        # 根据归一化异质性计算采样数
        hetero = hetero_dict_norm[seg_id]
        n_samples = int(base_samples + hetero * (max_samples - base_samples))
        # 随机采样
        selected_indices = np.random.choice(range(len(coordinates)), size=n_samples,replace=False)
        # 标记mask
        for idx in selected_indices:
            y, x = coordinates[idx]
            mask[y, x] = 1

    return mask


if __name__ == '__main__':
    out_shp_file = r'D:\Programing\pythonProject\St_Analyse\Position_mask\research1_init_mask.shp'

    img = Hyperspectral_Image()
    img.init(r"D:\Data\yanjiuqu\预处理\crop_for_research.dat")  # 使用原始数据的增强影像
    print(f'原始像素数量：{img.rows * img.cols}')
    # dataset = img.get_dataset()
    img.image_enhance(n_components=3) # 降维

    """超像素分割"""
    segments = superpixel_segmentation(img.enhance_img, n_segments=2048, compactness=10)
    print("超像素数量:", len(np.unique(segments)))
    """生成超像素颜色填充图"""
    plot_superpixel_figure(img.enhance_img, segments)

    # """随机取样"""
    if os.path.exists(out_shp_file):
        raise ValueError('文件已存在，重新指定文件名')
    hetero_dict = calculate_spectral_heterogeneity(img.get_dataset(), segments) # 计算光谱异质性
    mask = generate_adaptive_mask(segments, hetero_dict, base_samples_ratio=0.003, max_samples_ratio=0.005) # 训练模型建议两个值一样
    mask_to_vector_gdal(mask, img.dataset.GetGeoTransform(),img.dataset.GetProjection(),output_shapefile=out_shp_file)
    print(f"采样数量为：{np.sum(mask)}")
