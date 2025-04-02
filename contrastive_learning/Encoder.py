import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
import torch
from cnn_model.Encoder import SpectralSpatialCNN_3d,SpectralSpatialCNN


if __name__=='__main__':
    x = torch.randn(2,1,25,25,138)
    model = SpectralSpatialCNN_3d(20)
    y = model(x)
    print(y)