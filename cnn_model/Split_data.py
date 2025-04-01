import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
'''保证项目迁移能够正确导包'''

import random
import numpy as np
from base_utils.utils import read_txt_to_list,write_list_to_txt
def split_single_class_to_train_and_eval(dataset:list, ratio):
    """
    根据给定的比例将两个列表分为训练集和验证集(单个类)
    Parameters:
    data_list (list): 数据列表
    label_list (list): 标签列表
    ratio (float): 训练集的比例，默认 0.8
    Returns:
    tuple: (train_data, train_labels, val_data, val_labels)
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_size = int(len(dataset) * ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    return train_data, val_data,

def split_dataset_to_train_and_eval(dataset:list, ratio):
    """
    根据给定的比例将两个列表分为训练集和验证集(全部数据)，每个类别均已相同的ratio划分
    Parameters:
    data_list (list): 数据列表
    label_list (list): 标签列表
    ratio (float): 训练集的比例，默认 0.8
    Returns:
    tuple: (train_data, train_labels, val_data, val_labels)
    """
    label = [data.split()[1] for data in dataset]
    label = np.array(label,dtype=np.int16)
    dataset = np.array(dataset)
    classes = np.unique(label)
    train_data_lists,eval_data_lists = [],[]
    for c in classes:
        mask = (label==c)
        dataset_list = dataset[mask].tolist()
        train_data, eval_data = split_single_class_to_train_and_eval(dataset_list,ratio)
        train_data_lists += train_data
        eval_data_lists += eval_data
    return train_data_lists,eval_data_lists

if __name__ == '__main__':
    datasets = read_txt_to_list('enhance_data/.enhance_datasets.txt')

    train_data_lists, eval_data_lists = split_dataset_to_train_and_eval(datasets, 0.8)
    write_list_to_txt(train_data_lists, './split_dataset/train_datasets.txt')
    write_list_to_txt(eval_data_lists, './split_dataset/eval_datasets.txt')