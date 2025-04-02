import numpy as np
'''颜色条'''
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
colors = [
    '#C0E2FD', '#FEC0C1', '#CDC6FF', '#FDC0F7', '#F3D8F1',
    '#D6EBBF', '#E1CAF7', '#BFDCE2', '#F8F0BE', '#BEEFBF',
    '#F8C9C8', '#C0E2D2', '#E9BFC0', '#E3E3E3', '#BFBFBF',
    '#DEECF6', '#AFCBE2', '#E2F2CD', '#B6DAA7', '#F9D5D5',
    '#EF9BA1', '#FBE3C0', '#FBC99A', '#EBE0EF', '#C2B1D7',
]
def label_to_rgb(t):
    '''根据颜色条将label映射到rgb图像'''
    H, W = t.shape
    t=t.reshape(-1)
    rgb=[VOC_COLORMAP[i] for i in t ]
    rgb=np.array(rgb,dtype=np.uint8)
    rgb=rgb.reshape(H,W,3)
    return rgb
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