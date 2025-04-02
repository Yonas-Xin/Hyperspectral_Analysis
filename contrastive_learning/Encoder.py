import torch
from cnn_model.Encoder import SpectralSpatialCNN_3d,SpectralSpatialCNN


if __name__=='__main__':
    x = torch.randn(2,1,25,25,138)
    model = SpectralSpatialCNN_3d(20)
    y = model(x)
    print(y)