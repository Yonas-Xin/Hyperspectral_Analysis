import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
'''保证项目迁移能够正确导包'''


import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm  # 添加进度条
from Models import CNN_3d
from multiprocessing import cpu_count
from Data import Moni_leaning_dataset,MoniHDF5_leaning_dataset
from torch.optim.lr_scheduler import StepLR,ExponentialLR,ReduceLROnPlateau


from base_utils.utils import read_txt_to_list
from base_utils.Dataloader_X import DataLoaderX

# 学习率衰减：
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True) 按照验证损失率自动衰减
# scheduler = ExponentialLR(optimizer, gamma=0.9) 按照指数衰减
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 按照固定步长衰减
def print_info(mode=True):
    if mode == 1:
        cpu_num = cpu_count()  # 自动获取最大核心数目
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num)
        print('Using cpu core num: ', cpu_num)
    '''输出硬件信息'''
    print('Is cuda availabel: ', torch.cuda.is_available())  # 是否支持cuda
    print('Cuda device count: ', torch.cuda.device_count())  # 显卡数
    print('Current device: ', torch.cuda.current_device())  # 当前计算的显卡id
def get_systime():
    return datetime.now().strftime("%Y%m%d%H%M") # 记录系统时间

# 打印 GPU 设备信息
def print_gpu_info():
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if __name__ == '__main__':
     epochs = 300
     batch = 4
     init_lr = 1e-4
     min_lr = 1e-7
     current_script_path = os.getcwd()  # 获取当前运行脚本的目录
     model = CNN_3d(24,8)
     config_model_name = "CNN_3d"  # 模型名称
     # pretrain_model_name = './models/F3FN_retrain_202503291355.pth'  # 预训练模型
     # state_dict = torch.load(pretrain_model_name, weights_only=True)
     # model.load_state_dict(state_dict)

     # GPU 设置
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print_gpu_info()

     '''数据集及优化器'''
     dataset_train = 'train_datasets.h5'
     dataset_eval = 'eval_datasets.h5'
     optimizer = optim.Adam(model.parameters(), lr=init_lr)  # 优化器
     scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
     dataset_train = MoniHDF5_leaning_dataset(dataset_train)
     dataset_test = MoniHDF5_leaning_dataset(dataset_eval)
     dataloader_train = DataLoaderX(dataset_train, batch_size=batch, pin_memory=True, shuffle=True, num_workers=4)
     dataloader_test = DataLoaderX(dataset_test, batch_size=batch, pin_memory=True, shuffle=False, num_workers=4)
     Loss = nn.CrossEntropyLoss()
     device = torch.device('cuda')
     model.to(device)
     print_info(True)

     '''输出模型与日志'''
     current_time = get_systime()
     output_name = config_model_name + '_' + current_time  # 模型输出名称
     log = open(os.path.join(current_script_path, 'logs\\' + output_name + '.log'), 'w')
     model_name = os.path.join(current_script_path, 'models\\' + output_name + ".pth")

     '''训练策略配置'''
     train_epoch_best_loss = 99  # 初始化最佳loss
     no_optim = 0  # 用来记录loss不降的轮数
     max_no_optim_num = 1  # 当no_optim超过此值，更新学习率
     lr_rate = 3  # 学习率等比下降比例
     for epoch in range(epochs):
         start = time.time()
         model.train()
         running_loss = 0.0
         correct = 0
         for data, label in tqdm(dataloader_train, desc='Training:', total=len(dataloader_train)):
             data, label = data.to(device).unsqueeze(1), label.to(device)
             optimizer.zero_grad()
             output = model(data)
             loss = Loss(output, label)
             loss.backward()
             optimizer.step()
             running_loss += loss.item()
             _, predict = torch.max(output,1)
             correct += (predict == label).sum().item()
         end = time.time()
         avg_loss = running_loss / len(dataloader_train)
         accuracy = 100 * correct / len(dataset_train)
         current_lr = optimizer.param_groups[0]['lr']
         result = f"Epoch-{epoch + 1} , Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Lr: {current_lr:.8f}, time: {(end - start):.2f}"
         print(result)
         log.write(result+'\n')


         if dataloader_test is not None:
             model.eval()
             correct = 0
             running_loss = 0.0
             with torch.no_grad():
                 for data, label in tqdm(dataloader_test, desc='Testing', total=len(dataloader_test)):
                     data, label = data.to(device).unsqueeze(1), label.to(device)
                     output = model(data)
                     loss = Loss(output, label)
                     running_loss += loss.item()
                     _,predict = torch.max(output,1)
                     correct += (predict==label).sum().item()
                 avg_loss = running_loss / len(dataloader_test)
                 accuracy = 100 * correct / len(dataset_test)
                 result = f"Test_Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                 print(result)
                 log.write(result+'\n')
         if avg_loss >= train_epoch_best_loss:  # 若当前epoch的loss大于等于之前最小的loss
             no_optim += 1
         else:  # 若当前epoch的loss小于之前最小的loss
             no_optim = 0  # loss未降低的轮数归0
             train_epoch_best_loss = avg_loss  # 保留当前epoch的loss
             state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch}
             torch.save(state, model_name)
             print(f"模型参数、优化器参数已保存：{model_name}")
         if (epoch+1) > 200 or current_lr<=min_lr:
             pass
         else:
             scheduler.step()
         log.flush()