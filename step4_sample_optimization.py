from base_utils.core import Hyperspectral_Image,pca
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from base_utils.plotly_draw import show_scatter3d,create_scatter_3d,multiplot_scatter_3d
'''颜色条'''
VOC_COLORMAP = [[255, 255, 255], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
def gmm_cluster(data, mask, n_components=12):
    '''gmm聚类,返回具有标签的mask'''
    model = GaussianMixture(n_components=n_components)
    labels = model.fit_predict(data) + 1  # 不能让标签为0
    mask[mask > 0] = labels  # 样本类型更新
    return mask

def kmeans_cluster(data, mask, n_components=12):
    model = KMeans(n_clusters=n_components)
    labels = model.fit_predict(data) + 1  # 不能让标签为0
    mask[mask > 0] = labels  # 样本类型更新
    return mask

def load_deep_feature(path):
    dataset = np.load(path)
    return dataset['data']

if __name__ == '__main__':
    """数据读取，数据增强，样本随机选择"""
    if_plot = True
    img = Hyperspectral_Image()
    input_shp_file = r'D:\Programing\pythonProject\St_Analyse\Position_mask\research1_init_deep20_gmm_20.shp'
    deep_feature_npz_file = '../datasets/Deep_30Feature_FS3Ndrop_research1_init8220.npz'
    out_shp_file = r'..\Position_mask\research1_deep20_gmm24_samples_optimization_8.shp'

    img.init(r"D:\Data\yanjiuqu\预处理\crop_for_research.dat") # 加载原始影像
    mask = img.create_mask(input_shp_file) # 加载样本点位，如果将所有像素视作样本点则调用img.backward_mask()
    dataset = load_deep_feature(deep_feature_npz_file)
    mask = gmm_cluster(dataset,mask,24)
    print(len(dataset))

    idx = mask[mask > 0]    # 如果是8225数据改成 idx = mask[mask ！= 0]
    print(f"原始样本数量：{len(idx)}")

    if if_plot:
        pca_data = pca(dataset, 3)
        min = np.min(pca_data, axis=0)
        max = np.max(pca_data, axis=0)
        x_range, y_range, z_range = [min[0], max[0]], [min[1], max[1]], [min[2], max[2]]
        # figs = [create_scatter_3d(pca_data, idx, x_range=x_range, y_range=y_range, z_range=z_range, show_legend=True)]
        show_scatter3d(create_scatter_3d(pca_data, idx, show_connections=True, show_ellipsoid=False,  x_range=x_range, y_range=y_range, z_range=z_range, show_legend=True))
    classes = 20
    for j in range(8):
        for i in range(classes):
            label_mask = (idx==(i+1))
            data = dataset[label_mask]
            if data.shape[0]<75:
                continue
            label = np.zeros(data.shape[0])+(i+1)
            model = IsolationForest(n_estimators=100,
                                    max_samples='auto',
                                    contamination=float(0.1),
                                    max_features=1.0)
            x = model.fit_predict(data)
            label[x==-1] = -1
            idx[idx==(i+1)] = label # 孤立点位置标记为-1
        print(f"浓缩样本数量：{len(idx[idx > 0])}")
        if if_plot and j==7:
            # figs.append(create_scatter_3d(pca_data,idx,x_range=x_range, y_range=y_range, z_range=z_range, show_legend=True))
            show_scatter3d(create_scatter_3d(pca_data,idx,show_connections=True, show_ellipsoid=False,x_range=x_range, y_range=y_range, z_range=z_range, show_legend=True))
        mask[mask != 0] = idx # 更新mask
    img.create_vector(mask, out_shp_file)