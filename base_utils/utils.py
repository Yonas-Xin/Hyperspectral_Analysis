import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

    
def search_files_in_directory(directory, extension):
    """
    搜索指定文件夹中所有指定后缀名的文件，并返回文件路径列表,只适用于不需要标签训练的模型，因为返回的列表顺序可能和
    需要的顺序不同，使用需慎重，但是同一命名规则返回的列表一定是相同的
    Parameters:
        directory (str): 要搜索的文件夹路径
        extension (str): 文件后缀名，应该以 '.' 开头，例如 '.txt', '.jpg'
    Returns:
        list: 包含所有符合条件的文件路径的列表
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                matching_files.append(os.path.join(root, file))
    return matching_files
def read_txt_to_list(filename):
    with open(filename, 'r') as file:
        # 逐行读取文件并去除末尾的换行符
        data = [line.strip() for line in file.readlines()]
    return data

def write_list_to_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")  # 每个元素后加上换行符
        file.flush()