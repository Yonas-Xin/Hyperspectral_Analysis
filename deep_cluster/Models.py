import torch
import torch.nn as nn
import torch.nn.functional as F


class DEC(nn.Module):
    '''编码器从其他模型获取'''
    def __init__(self, encoder, n_clusters, encoder_out_features, a=1):
        super().__init__()
        self.encoder = encoder # 预训练的编码器
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, encoder_out_features)) # 聚类参数，每一行代表一个聚类中心
        self.a = a # 控制距离度量时各个特征的度数

    def forward(self, spectral, space):
        '''
        注意这里要将所有样本一次性输入，batch设置为最大
        spectral: (num_simples, spectral_dims)
        space: (num_simples, block, block, space_channels)
        '''
        space = self.encoder(spectral, space) # 嵌入表示
        # 计算欧式距离相似性矩阵
        q = 1.0 + torch.sum(torch.pow(space.unsqueeze(1) - self.cluster_layer, 2), 2) / self.a # 欧式距离平方
        q = q.pow(-(self.a + 1.0) / 2.0) # 幂运算
        q = q/torch.sum(q, 1, keepdim=True) # 归一化,软分配
        return q