import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import os.path
from osgeo import gdal,ogr,osr
import numpy as np
from threading import Lock
from tqdm import tqdm
gdal_lock = Lock()
nodata_value = 0

def write_data_to_tif(output_file, data, geotransform, projection, nodata_value=nodata_value, dtype=gdal.GDT_Float32):
    """
    将数据写入 GeoTIFF 文件，并保留与原文件相同的元数据
    Parameters:
        output_file: str, 输出的 GeoTIFF 文件路径
        data: ndarray, 写入的数据，形状为(row, col, bands)
        geotransform: tuple, GeoTIFF 的地理变换参数
        projection: str, WKT 格式的投影信息
        dtype: gdal 数据类型, 默认 gdal.GDT_Float32
    Returns:
        None
    """
    bands, rows, cols =data.shape
    data = data.astype(np.float32)
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_file, cols, rows, bands, dtype)
    if dataset is None:
        raise IOError(f"无法创建文件 {output_file}")
    # 设置地理变换和投影
    if geotransform is not None and projection is not None:
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(projection)
    # 写入数据
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(data[i,:,:])
        band.SetNoDataValue(nodata_value)  # 设置 NoData 值
    # 释放资源
    dataset.FlushCache()
    dataset = None
    return output_file

def read_tif_with_gdal(tif_path):
    '''读取栅格原始数据(形状为 bands x rows x cols)'''
    gdal.UseExceptions()
    with gdal_lock:
        dataset = gdal.Open(tif_path)
        dataset = dataset.ReadAsArray()
    return dataset
def crop_image_by_mask(data, mask, geotransform, projection, filepath, block_size=30, name="Block_", dtype = gdal.GDT_Float32):
    """
    根据 mask 的类别，裁剪影像为 30x30 的小块
    :param data: 输入影像，(C, H, W)
    :param mask: 二维 mask，形状为 (rows, cols)，背景为0，其他为类别
    :param block_size: 每个块的大小，默认为30
    :return: ndarray,(nums,block_size,block_size,bands)
    """
    bands, rows, cols = data.shape
    if block_size%2 == 0:#如果block_size是一偶数，以像素点为中心，左上角区域比右下角区域少一
        left_top = int(block_size/2-1)
        right_bottom = int(block_size/2)
    else:
        left_top = int(block_size//2)
        right_bottom = int(block_size//2)
    data = np.pad(data,[(0,0),(left_top,right_bottom),(left_top,right_bottom)],'constant')
    num = 1
    pathlist = []
    add_labels = False
    # 遍历 mask 的每个像素
    if np.max(mask)>1: # 如果大于1说明裁剪的图像有标签
        print('有标签，将额外生成标签至txt文件')
        add_labels = True
    else: print('无标签，生成纯数据地址txt文件')
    for row in tqdm(range(rows),desc='Cropping', total=rows):
        for col in range(cols):
            if mask[row, col] > 0:  # 如果该位置不是背景和噪声
                path = os.path.join(filepath, name + str(num) + ".tif")
                # 计算裁剪图像的左上角坐标（originX, originY）
                originX = geotransform[0] + (col-left_top) * geotransform[1]
                originY = geotransform[3] + (row-left_top) * geotransform[5]
                # 计算新的 GeoTransform
                new_geotransform = (originX, geotransform[1], geotransform[2], originY, geotransform[4], geotransform[5])
                block = data[:, row:row + block_size, col:col + block_size]
                write_data_to_tif(path, block, geotransform=new_geotransform, projection=projection, dtype=dtype)
                num += 1
                if add_labels:
                    pathlist.append(path + f' {mask[row, col]-1}')
                else: pathlist.append(path)
    dataset_path = os.path.join(filepath, '.datasets.txt')
    write_list_to_txt(pathlist, dataset_path)
    # if np.max(mask)>1: # 如果大于1说明裁剪的图像有标签
    #     label_path = os.path.join(filepath, '.labels.txt')
    #
    #     write_list_to_txt(labels,label_path) # 裁剪的文件夹下创建label txt文件
    #     write_list_to_txt(pathlist, dataset_path) # 裁剪的文件夹下创建dataset txt文件
    #     print(f'样本路径下生成了标签文件：{label_path}')
    #     print(f'样本路径下生成了数据集文件：{dataset_path}')
    # else: # 生成无标签的样本集
    #     dataset_path = os.path.join(filepath, '.datasets.txt')
    #     write_list_to_txt(pathlist, dataset_path) # 裁剪的文件夹下创建dataset txt文件
    #     print(f'样本路径下生成了数据集文件(无标签)：{dataset_path}')

def write_list_to_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")  # 每个元素后加上换行符
        file.flush()

def create_dataset_from_file(filepath, extension = '.tif'):
    images = search_files_in_directory(filepath, extension = extension)
    return images

def search_files_in_directory(directory, extension='.txt'):
    """
    搜索指定文件夹中所有指定后缀名的文件，并返回文件路径列表
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

def mask_to_vector_gdal(mask_matrix,geotransform, projection=None, output_shapefile="./Position_mask/test.shp"):
    """
    将二维矩阵mask矩阵转化为矢量点文件
    """
    # 获取矩阵的行和列
    rows, cols = mask_matrix.shape
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if not driver:
        raise RuntimeError("Shapefile driver not available")
    data_source = driver.CreateDataSource(output_shapefile)
    # 创建一个图层，用于存储点（Point）几何
    spatial_ref = osr.SpatialReference()
    if projection:
        spatial_ref.ImportFromWkt(projection) # 定义投影坐标系
    layer = data_source.CreateLayer('points', geom_type=ogr.wkbPoint, srs=spatial_ref)
    field = ogr.FieldDefn('class', ogr.OFTInteger)
    layer.CreateField(field)
    # 遍历矩阵，提取非零值的坐标并创建点特征
    for row in range(rows):
        for col in range(cols):
            value = mask_matrix[row, col]
            if value > 0:  # 非零值表示分类
                # 创建一个点
                geo_x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
                geo_y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
                point = ogr.Geometry(ogr.wkbPoint)
                value = int(value-1)
                point.AddPoint(geo_x, geo_y)
                # 创建一个要素（Feature）并设置几何和属性值
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetGeometry(point)
                feature.SetField('class', value)  # 设置分类属性值
                layer.CreateFeature(feature)  # 将特征写入图层
                feature = None  # 清理
    # 关闭数据源，保存Shapefile
    data_source = None
    return f"shp文件已保存，文件地址：{output_shapefile}"


def vector_to_mask(shapefile, geotransform, rows, cols):
    """
    将矢量点文件转化为二维矩阵（mask矩阵）
    """
    mask_matrix = np.zeros((rows, cols), dtype=int)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = ogr.Open(shapefile)
    if not data_source:
        raise RuntimeError(f"Failed to open shapefile: {shapefile}")
    layer = data_source.GetLayer()
    spatial_ref = layer.GetSpatialRef()
    if spatial_ref is None:
        raise RuntimeError("No spatial reference found in shapefile")
    # 处理每个要素（点）
    for feature in layer:
        # 获取点的几何体
        geometry = feature.GetGeometryRef()
        if geometry.GetGeometryType() != ogr.wkbPoint:
            continue  # 只处理点类型的几何
        geo_x, geo_y = geometry.GetX(), geometry.GetY()
        col = int((geo_x - geotransform[0]) / geotransform[1])  # 计算列索引
        row = int((geo_y - geotransform[3]) / geotransform[5])  # 计算行索引
        if 0 <= row < rows and 0 <= col < cols:        # 确保索引在矩阵范围内
            # 获取'Class'字段的值，假设在Shapefile中为'class'
            value = feature.GetField('class')
            mask_matrix[row, col] = value + 1  # 假设从1开始的class，且需要恢复为原始值
    data_source = None
    return mask_matrix

def vector_to_raster_matrix_gdal(raster_dataset, vector_path, attribute="class"):
    """
    使用GDAL将矢量文件转换为与栅格一致的NumPy矩阵，值为矢量面属性（class），未覆盖区域填充0
    :param raster_path: 参考栅格文件路径
    :param vector_path: 矢量文件路径（Shapefile）
    :param attribute: 矢量属性字段名称（默认 'class'）
    :return: NumPy矩阵 (H, W)
    """
    geotransform = raster_dataset.GetGeoTransform()  # 仿射变换
    projection = raster_dataset.GetProjection()  # 投影信息
    width = raster_dataset.RasterXSize  # 栅格宽度
    height = raster_dataset.RasterYSize  # 栅格高度

    # 2. 打开矢量文件
    vector_ds = ogr.Open(vector_path)
    if not vector_ds:
        raise RuntimeError(f"无法打开矢量文件 {vector_path}")

    layer = vector_ds.GetLayer()

    # 3. 创建一个内存栅格数据集，存储矢量化数据
    mem_driver = gdal.GetDriverByName("MEM")
    mem_raster = mem_driver.Create("", width, height, 1, gdal.GDT_Int16)
    mem_raster.SetGeoTransform(geotransform)  # 设定仿射变换
    mem_raster.SetProjection(projection)  # 设定投影

    band = mem_raster.GetRasterBand(1)
    band.Fill(0)  # 其他区域填充 0
    gdal.RasterizeLayer(mem_raster, [1], layer, burn_values = [1]) # 矢量覆盖区填充1
    matrix = band.ReadAsArray()
    vector_ds = None
    mem_raster = None

    return matrix

def point_value_merge(shapefile, value:list):
    """
    要素属性修正
    """
    base_value = value[0]
    merge_value = value[1:]
    data_source = ogr.Open(shapefile, 1)
    if not data_source:
        raise RuntimeError(f"Failed to open shapefile: {shapefile}")
    try:
        layer = data_source.GetLayer()
        modified_count = 0
        all_count = 0

        # 遍历所有要素
        for feature in layer:
            # 验证几何类型为点
            all_count += 1
            geometry = feature.GetGeometryRef()
            if geometry and geometry.GetGeometryType() != ogr.wkbPoint:
                continue

            # 获取字段值
            field_value = feature.GetField('class')

            # 执行属性修改
            if field_value in merge_value:
                feature.SetField('class', base_value)
                layer.SetFeature(feature)  # 提交修改
                modified_count += 1

        print(f'要素属性修正完成，共修改 {modified_count} 个点要素，文件总要素{all_count}')

    finally:
        # 确保资源释放
        data_source.Destroy()
        data_source = None
