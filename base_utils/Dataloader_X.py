from prefetch_generator import BackgroundGenerator  # pip install prefetch_generator
from torch.utils.data import DataLoader

class DataLoaderX(DataLoader):
    """(加速组件) 重新封装Dataloader，使prefetch不用等待整个iteration完成"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())