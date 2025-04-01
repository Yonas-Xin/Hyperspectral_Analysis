'''
根据样本.shp文件进行样本的裁剪
'''
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from base_utils.gdal_utils import crop_image_by_mask,write_list_to_txt,vector_to_mask
from base_utils.core import *


def concrete_sample_name(dataset_paths1, dataset_paths2):
    '''将两个列表元素写入txt'''
    label_with_dataset_all = []
    for i in range(len(dataset_paths2)):
        filename1 = dataset_paths1[i]
        filename2 = dataset_paths2[i]
        label_with_dataset = filename1 + " " + filename2
        label_with_dataset_all.append(label_with_dataset)
    return label_with_dataset_all

def create_block_sample(img, filepath_dir,block_size=30):
    '''将数据按照mask标签进行样本裁剪，同时裁剪光谱样本和增强影像样本，生成样本路径及标签的txt文件'''
    block_path = os.path.join(filepath_dir, 'block')
    spectral_path = os.path.join(filepath_dir, "spectral")
    if not os.path.exists(block_path):
        os.mkdir(block_path)
        os.mkdir(spectral_path)
    else:
        raise ValueError('文件夹已存在，为避免文件夹覆盖，请重新设置文件夹名称或者删除已存在的文件夹')
    crop_image_by_mask(img.enhance_img.transpose(2,0,1), img.sampling_position, img.dataset.GetGeoTransform(),
                                  img.dataset.GetProjection(),filepath=block_path, block_size=block_size, name = "space_")
    crop_image_by_mask(img.get_dataset().transpose(2,0,1), img.sampling_position, img.dataset.GetGeoTransform(),
                                  img.dataset.GetProjection(),filepath=spectral_path, block_size=1, name = "spectral_")
    print("数据裁剪成功")

def create_block_sample_3d(img, filepath_dir, name='block3d_', block_size=30):
    '''裁剪适合3d 输入的影像块'''
    if not os.path.exists(filepath_dir):
        os.mkdir(filepath_dir)
    else:
        raise ValueError('文件夹已存在，为避免文件夹覆盖，请重新设置文件夹名称或者删除已存在的文件夹')
    crop_image_by_mask(img.get_dataset().transpose(2,0,1), img.sampling_position, img.dataset.GetGeoTransform(),
                                  img.dataset.GetProjection(),filepath=filepath_dir, block_size=block_size, name = name)

if __name__ == "__main__":
    img = Hyperspectral_Image()
    area_data = r'D:\Data\yanjiuqu\预处理\Area2_for_sampling.dat' # 裁剪区域栅格影像
    label_mask = r"C:\Users\85002\Desktop\毕设\Area2_mask.shp" # 裁剪shp文件
    dir_name = 'block_clip_for_contrastive_learning2' # 设置一个目录存放裁剪的数据
    current_dir = os.getcwd()
    out_dir = os.path.join(current_dir, dir_name)

    img.init(area_data)
    mask = vector_to_mask(label_mask, img.dataset.GetGeoTransform(),img.rows,img.cols)
    print(f"样本标记数据量：{np.sum(mask>0)}")
    if np.sum(mask) < 100000:
        img.sampling_position = mask
        create_block_sample_3d(img, out_dir, name='Area2_', block_size=25) # 裁剪成块,使用绝对路径
    else:
        print("数据量限制十万")