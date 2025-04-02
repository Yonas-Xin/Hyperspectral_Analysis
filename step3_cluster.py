'''
从分类结果上看，高斯混合模型的分类结果较KMeans结果较优，Ncut模型分类器选择高斯混合模型
'''

import numpy as np
from base_utils.core import Hyperspectral_Image
import numpy as np
import os
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from base_utils.gdal_utils import mask_to_vector_gdal, vector_to_mask
from sklearn.preprocessing import normalize
'''颜色条'''
VOC_COLORMAP = [[255, 255, 255], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
def load_data(filename):
    """
    载入数据
    :param filename: 文件名
    :return:
    """
    data = np.loadtxt(filename, delimiter='\t')
    return data
def compute_euclidean_distance(X):
    '''计算所有样本点之间的欧式距离(xi-xj）^2'''
    X_norm = np.sum(X**2, axis=1)  # 每个样本的L2范数平方
    D_sq = X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T) #（xi-xj）^2
    D_sq = np.maximum(D_sq, 0)  # 避免因浮点误差导致负数
    return D_sq
def compute_rbf_similarity(X,gamma=None):
    '''计算高斯核函数相似矩阵'''
    D_sq = compute_euclidean_distance(X)
    if gamma is None: # 以中位数设置gamma
        distances = np.sqrt(D_sq + 1e-8)  # 添加小常数防止除零
        sigma = np.median(distances[distances > 1e-8])  # 过滤掉自身距离,中位数
        gamma = 1.0 / (2 * sigma**2)  # 默认gamma公式
    S = np.exp(-gamma * D_sq)
    np.fill_diagonal(S, 1.0)  # 将对角线设为1（样本与自身相似度为1）
    return S
def Z_Score(data, axis=0):
    '''标准化'''
    mean = np.mean(data,axis=axis, keepdims=True)    # 均值
    std_dev = np.std(data,axis=axis, keepdims=True)    # 标准差
    norm_data = (data - mean) / std_dev
    return norm_data
def N_cut(X,n_components=20,n_cluster=12,max_iter=1000, gamma=None):
    '''
    N_cut算法
    :param X: （nums，features）输入原始数据
    :param n_components: 降维后数据维度
    :param n_cluster: 分类数
    :param max_iter: 模型迭代次数，当出现模型未拟合警告时，调大该值
    :return: label（nums，）
    '''
    W = compute_rbf_similarity(X,gamma=gamma) # 相似矩阵，作为邻接矩阵
    D = np.diag(np.sum(W,axis=1)) # 度矩阵
    L = D-W # 拉普拉斯矩阵
    D = np.linalg.inv(np.sqrt(D)) # D^-1/2
    L = D @ L @ D # 归一化拉普拉斯矩阵

    eigenvalues, eigenvectors = np.linalg.eigh(L)  # 特征分解
    sorted_idx = np.argsort(eigenvalues)  # 升序,选择小的特征值
    eigenvectors_sorted = eigenvectors[:, sorted_idx]
    f = Z_Score(eigenvectors_sorted[:, :n_components], axis=1) # 每行进行标准化
    return GaussianMixture(n_components=n_cluster, n_init=20).fit_predict(f) # 使用高斯混合模型作为聚类方法

def label_to_rgb(label_feature, map=VOC_COLORMAP, transpose_to_int=True):
    '''
    根据颜色条将label映射到rgb图像
    :param label_feature: （rows，cols）
    :param map: 颜色条
    :param transpose_to_int: 是否转化为int类型
    :return:
    '''
    H, W = label_feature.shape
    label_feature=label_feature.reshape(-1)
    rgb=[map[i] for i in label_feature]
    rgb=np.array(rgb,dtype=np.uint8)
    if transpose_to_int:
        rgb =rgb.astype(np.uint8)
    else:
        rgb = rgb.astype(np.float32)
    rgb=rgb.reshape(H,W,3)
    return rgb

def bic_score(data, k_range=30):
    """贝叶斯准则,分数最小的k值为最佳分类数"""
    bic_scores = []
    for k in range(k_range):
        gmm = GaussianMixture(n_components=k)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
        print(gmm.bic(data))
    bic_scores = np.array(bic_scores)
    print(f'最小分数：{np.min(bic_scores)} k值：{np.argmin(bic_scores)}')
    plt.plot(k_range, bic_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('BIC Score')
    plt.show()

def load_deep_feature(path):
    dataset = np.load(path)
    return dataset['data']

if __name__ == '__main__':
    """数据读取，数据增强，样本随机选择"""
    img = Hyperspectral_Image()
    input_shp_file = r'D:\Programing\pythonProject\St_Analyse\Position_mask\research1_init_mask.shp'
    deep_feature_npz_file = '../datasets/Deep_30Feature_FS3Ndrop_research1_init8220.npz'
    out_shp_file = r'..\Position_mask\research1_init_deep30_ncut_20.shp'


    img.init(r"D:\Data\yanjiuqu\预处理\crop_for_research.dat") # 加载原始影像
    mask = img.create_mask(input_shp_file)
    print(np.sum(mask))
    dataset = load_deep_feature(deep_feature_npz_file) # 加载深度特征

    labels = N_cut(dataset,n_components=20, n_cluster=20, gamma=None, ) + 1 # 分类

    # model = GaussianMixture(n_components=20, n_init=10)
    # labels = model.fit_predict(dataset) + 1 #不能让标签为0
    mask[mask != 0] = labels  # 样本类型更新
    img.create_vector(mask,out_shp_file)