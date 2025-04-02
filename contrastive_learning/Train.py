import os
from pathlib import Path
from Data import SSF,SSF_3D,SSF_3D_H5
from base_utils.Dataloader_X import DataLoaderX
import torch
import torch.optim as optim
from time import time
from datetime import datetime
from Models import F3FN,F3FN_3d
from Feature_transform import BatchAugment,add_gaussian_noise_torch,BatchAugment_3d
from loss import InfoNCELoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR,ExponentialLR,StepLR
from tqdm import tqdm
from multiprocessing import cpu_count
from base_utils.utils import search_files_in_directory
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

def join_path(path1, path2):
    return os.path.join(path1, path2)

if __name__ =="__main__":
    """基础配置"""
    epochs = 300  # epoch
    batch = 2    # batch
    init_lr = 1e-5    # lr
    min_lr = 1e-7
    learning_mode = 0 # 设置为1适合模型调整，0适合模型初期训练
    current_script_path = Path(__file__).parent # 获取当前运行脚本的目录
    config_model_name = "F3FN_3d_pretrain"   # 模型名称
    # pretrain_model_name = './models/F3FN_retrain_202503291355.pth' # 预训练模型

    '''模型和优化器'''
    model = F3FN_3d(24)
    # state_dict = torch.load(pretrain_model_name, weights_only=True)
    # model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters() ,lr = init_lr)   # 优化器
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=min_lr)
    full_cpu_mode = True  # 是否全负荷使用CPU，默认pytroch使用cpu一半核心

    '''数据集和增强模块'''
    current_time = get_systime()
    output_name = config_model_name +'_' + current_time  # 模型输出名称
    # image_paths = search_files_in_directory(r'D:\Programing\pythonProject\Hyperspectral_Analysis\data_process\block_clip','.tif')
    dataset = SSF_3D_H5('D:\Programing\pythonProject\data_store\contrastive_learning_138_25_25.h5')
    info_nce = InfoNCELoss(temperature=0.07)  # 损失函数
    dataloader = DataLoaderX(dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=4)  # 数据迭代器
    augment = BatchAugment_3d(flip_prob=0.5, rot_prob=0.5, gaussian_noise_std=(0.006, 0.012))  # 数据特征转换

    log = open(os.path.join(current_script_path, 'logs\\'+output_name+'.log'), 'w')
    model_name = os.path.join(current_script_path, 'models\\'+output_name + ".pth")
    device = torch.device('cuda')
    model.to(device)

    '''训练策略配置'''
    warm_up_epoch = 0  # 热身epoch
    train_epoch_best_loss = 99  # 初始化最佳loss
    no_optim = 0  # 用来记录loss不降的轮数
    max_no_optim_num = 1  # 当no_optim超过此值，更新学习率
    lr_rate = 3  # 学习率等比下降比例

    print_info(full_cpu_mode)
    for epoch in range(epochs):
        start = time()
        model.train()
        running_loss = 0.0
        for block in tqdm(dataloader, total=len(dataloader), desc="Train", leave=False):
            block = block.to(device).unsqueeze(1)
            block1 = augment(block)
            block2 = augment(block)
            optimizer.zero_grad()  # 清空梯度
            embedding, out = model(torch.cat((block1, block2), dim=0))

            # 掩膜噪声负样本
            # info_nce.cosine_similarity_matrix(embedding, th=0.9) # 第一次预训练时不要设置这个

            loss = info_nce(out)  # 对比损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()
        end = time()
        avg_loss = running_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        result = f"Epoch-{epoch + 1} , Loss: {avg_loss:.8f}, Lr: {current_lr:.8f}, time: {(end - start):.2f}"
        log.write(result + '\n')
        print(result)
        if avg_loss >= train_epoch_best_loss:  # 若当前epoch的loss大于等于之前最小的loss
            no_optim += 1
        else:  # 若当前epoch的loss小于之前最小的loss
            no_optim = 0  # loss未降低的轮数归0
            train_epoch_best_loss = avg_loss  # 保留当前epoch的loss
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_name)
            print(f"模型参数、优化器参数已保存：{model_name}")
        # 更新学习率
        if learning_mode == 1:
            if no_optim > 9:  # 若过多epoch后loss仍不下降则终止训练
                print(log, 'early stop at %d epoch' % epoch)  # 打印信息至日志
                print('early stop at %d epoch' % epoch)
                break
            if no_optim > max_no_optim_num:  # 多轮epoch后loss不下降则更新学习率
                if current_lr < min_lr:  # 当前学习率过低终止训练
                    break
                try:
                    model.load_state_dict(torch.load(model_name))  # 读取保存的loss最低的模型
                    print("重新加载最佳模型参数")
                except:
                    pass
                # 更新优化器的学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr / lr_rate
                no_optim = 0  # loss未降低轮数归0
                log_msg = f"Learning rate reduced to {current_lr / lr_rate}, reloaded best model."
                log.write(log_msg + '\n')
                print(log_msg)
        else:
            if (epoch + 1) < warm_up_epoch:
                pass
            else:
                scheduler.step()
        log.flush()