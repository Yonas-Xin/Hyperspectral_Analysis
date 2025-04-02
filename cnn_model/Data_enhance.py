import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
'''保证项目迁移能够正确导包'''

import numpy as np
from contrastive_learning.Feature_transform import BatchAugment_3d
from base_utils.utils import read_txt_to_list,write_list_to_txt
import os.path
from Data import Moni_leaning_dataset
from base_utils.Dataloader_X import DataLoaderX
from tqdm import tqdm
from base_utils import gdal_utils
def enhance_dataset(dataset_path_list, out_dir, factor=5, batch = 256):
    '''
    数据集扩充，扩充的数据形成tif文件，供检查
    :param dataset_path_list: 数据地址list
    :param labels_list: 数据label list
    :param factor: 扩充倍数
    :param batch: 扩充时的批量处理数
    :return: 无，创建原始数据集与增强数据集的地址和label
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        raise ValueError('文件夹已存在，为避免文件夹覆盖，请重新设置文件夹名称或者删除已存在的文件夹')

    base_name_lit = [(os.path.basename(dataset_path).split())[0] for dataset_path in dataset_path_list]
    dataset = Moni_leaning_dataset(dataset_path_list)
    dataloader = DataLoaderX(dataset, shuffle=False, batch_size=batch, num_workers=4)
    datasets_txt = os.path.join(out_dir, '.enhance_datasets.txt')

    datasets_txt_file = open(datasets_txt,'w')
    for path in dataset_path_list:
        datasets_txt_file.write(path+'\n') # 先复制一遍原始数据集
    augment = BatchAugment_3d()
    augment.generate_enhance_list(factor=factor)
    for i in range(factor):
        current_image_pos = 0
        for data,label in tqdm(dataloader, total=len(dataloader)):
            data = augment.order(data, idx = i).numpy()
            for j in range(data.shape[0]):
                tif_name = os.path.join(out_dir,f"EH_{i}_"+base_name_lit[current_image_pos])
                datasets_txt_file.write(tif_name+f' {label[j]}\n')
                gdal_utils.write_data_to_tif(tif_name, data[j], None,None)
                current_image_pos += 1
    datasets_txt_file.flush()

if __name__ == '__main__':
    pass
    enhance_out_dir_name = 'enhance_data'
    current_dir = os.getcwd()
    enhance_out_dir = os.path.join(current_dir, enhance_out_dir_name)

    image_paths = read_txt_to_list(r'..\data_process\block_clip\.datasets.txt')
    enhance_dataset(image_paths, enhance_out_dir,5 ,256) # 样本数据扩增