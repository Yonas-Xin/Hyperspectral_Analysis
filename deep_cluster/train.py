'''
深度聚类模型DEC的实现
'''
import os.path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from contrastive_learning import SSF
from time import time
from datetime import datetime
from Models import DEC
from  torch.nn.functional import kl_div
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR,ExponentialLR
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np

def set_parameter_requires_grad(model, feature_extracting):
    """
    是否保留梯度, 实现冻层
    :param model: 模型
    :param feature_extracting: 是否冻层
    :return: 无返回值
    """
    if feature_extracting:  # 如果冻层
        for param in model.parameters():  # 遍历每个权重参数
            param.requires_grad = False  # 保留梯度为False

def target_distribution(q):
    '''根据q计算目标分配p'''
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def split_spectral_from_txt(txt_file):
    image_paths = []
    spectrals = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            image_path, spectral = line.strip().split(' ')
            image_paths.append(image_path)
            spectrals.append(spectral)
    image_paths = np.array(image_paths)
    spectrals = np.array(spectrals)
    return image_paths,spectrals

if __name__ == '__main__':
    """config"""
    epochs = 300  # epoch
    init_lr = 0.000008     # lr
    config_model_name = "DEC"   # 模型名称
    # Encoder_model = F3FN(1,3,10)
    # state_dict = torch.load('../models/F3SN_20_Adam_pretrain_sample29000_202503231112.pth')
    # Encoder_model.load_state_dict(state_dict)
    model = DEC(None, 12, encoder_out_features=10)  # 模型
    Encoder_model = None
    model.encoder = Encoder_model.encoder
    del Encoder_model
    optimizer = optim.Adam(model.parameters(),lr = init_lr)   # 优化器

    current_time = datetime.now()
    current_time = current_time.strftime("%Y%m%d%H%M")
    output_name = config_model_name +current_time  # 模型输出名称
    image_paths,spectral_paths = split_spectral_from_txt(r".\data_txt\dataset_research3_10138.txt") # 加载数据路径
    dataset = SSF(image_paths, spectral_paths)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, pin_memory=True)  # 数据迭代器,一次性取出所有数据
    scheduler = ExponentialLR(optimizer, gamma=0.99)   # 学习率衰减策略,指数级衰减

    log_name = os.path.join('../logs','log'+output_name+'.txt')
    model_name = os.path.join('../models', output_name+".pth")
    log = open(log_name, 'w')
    loss_func = kl_div   # 损失函数
    device = torch.device('cpu')
    model.to(device)
    log_list = []
    with torch.no_grad():
        model.train()
        for space,spectral in dataloader:
            space, spectral = space.to(device), spectral.to(device)
            embedding = model.encoder(spectral, space)
    kmeans = KMeans(n_clusters=12, n_init=20)
    y_pred = kmeans.fit_predict(embedding.data.cpu().numpy())  # 使用 KMeans 算法对嵌入向量进行聚类，并获取聚类结果。
    y_pred_last = y_pred  #将最终聚类结果保存到变量 y_pred_last 中，用于后续判断停止条件。
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) #将聚类中心作为聚类层参数进行初始化。
    for epoch in range(epochs):
        start = time()
        model.train()
        for space,spectral in dataloader:
            space, spectral = space.to(device), spectral.to(device)
            if epoch == 0:
                q =model(spectral, space)
                p = target_distribution(q) # 获取目标分配概率
            optimizer.zero_grad()  # 清空梯度
            q = model(spectral, space)
            loss = loss_func(q.log(), p, reduction='batchmean') #reduction='batchmean' 表示将所有样本的 KL 散度平均成一个标量。
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
        end = time()
        result = f"Epoch-{epoch+1} , Loss: {loss.item():.7f}, Lr: {optimizer.param_groups[0]['lr']:.8f}, time: {(end-start):.2f}"
        log.write(result+'\t')
        print(result)
        # 更新学习率
        scheduler.step()
        if (epoch+1) % 20 ==0: # 每训练100批次，保存一次模型
            torch.save(model.state_dict(), model_name)
        log.flush()