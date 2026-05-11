import sys, os
sys.path.append('.')
from contrastive_learning.Models.Frame import Contrastive_Frame, train
from contrastive_learning.Models.Data import Contrastive_Dataset, MIRBS_Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from Models.Models import Ete_Model, Moco_Model
from contrastive_learning.Models.Feature_transform import HighDimBatchAugment
from utils import search_files_in_directory
from multiprocessing import cpu_count
from contrastive_learning.Models.Scheduler import WarmupLinearSchedule
TRAIN_DICT = {
    "ETE": Ete_Model,
    "MOCO": Moco_Model
}
DATASET_DICT = {
    1: Contrastive_Dataset,
    2: MIRBS_Dataset
}
# 可选用的模型如下：
# 'SRACN'  "Res_3D_18Net" "Res_3D_34Net" "Res_3D_50Net"
# 'HybridSN' 'Vgg16' 'MobileNetV1' 'MobileNetV2' 'ResNet18' 'ResNet34' 'ResNet50' 'spec_transformer'

encoder_model_name = 'spec_transformer' # 选择模型名称
config_model_name = "Test" # 配置后缀名
TRAIN_MODE = "ETE" # ETE or MOCO 训练方式选择
DATA_MANAGE_MODE = 1 # 数据管理方式，1为根据裁剪的样本进行训练，2为根据选择的影像自动训练
patch_size = 17 # DATA_MANAGE_MODE=2 时需指定该参数
images_dir = r'' # 数据集


if_full_cpu = True # 是否全负荷cpu
epochs = 10  # epoch, 实际训练轮数为 epochs + warm_up_epochs
batch = 12  # batch
init_lr = 1e-4  # lr
min_lr = 1e-6 # 最低学习率
warm_up_epochs = 10  # 预热epoch数
ck_pth = None
DISPLAY_NUMS = 1  # 每个epoch打印多少次训练信息，小于2不打印

K = 65536
m = 0.999
T = 0.07
MUITITHREADING_MODE = True # 是否使用多线程加速数据加载
USE_DATA_PARALLEL = True # 是否使用DataParallel进行多显卡训练

"""特征图绘制相关参数"""
FEATURE_MAP_LAYER_NAMES = None # 指定需要绘制特征图的层名，使用列表形式，例如 ['encoder','layer1.0.conv1']，为None则绘制所有层
FEATURE_MAP_NUM = 36 # 每个层绘制的特征图数量
FEATURE_MAP_INTERVAL = 10 # 每隔多少个epoch绘制一次特征图
if __name__ == '__main__':  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 显卡设置
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        USE_DATA_PARALLEL = False
    dataloader_num_workers = cpu_count() // 4 # 根据cpu核心数自动决定num_workers数量
    dataloader_num_workers = dataloader_num_workers if MUITITHREADING_MODE else 0
    # 配置dataloader
    image_lists = search_files_in_directory(images_dir, ('tif', 'dat'))
    dataset = DATASET_DICT[DATA_MANAGE_MODE](image_lists, patch_size=patch_size, multith_mode=MUITITHREADING_MODE)
    model = TRAIN_DICT[TRAIN_MODE](encoder_model_name=encoder_model_name, in_shape=dataset.data_shape, K=K, m=m, T=T)  # 模型实例化
    print(f"🔍 PyTorch Version: {torch.__version__}")
    print(f"🎯 Image shape: {dataset.data_shape}")
    optimizer = optim.Adam(model.parameters(), lr=init_lr)  # 优化器
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warm_up_epochs, t_total=epochs+warm_up_epochs, min_lr=min_lr)

    augment = HighDimBatchAugment(spectral_mask_prob=0.5, spectral_mask_p=0.25, band_dropout_prob=0.5, bands_dropout_p=0.25)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=dataloader_num_workers, prefetch_factor=2,
                            persistent_workers=MUITITHREADING_MODE and dataloader_num_workers > 0, drop_last=True)  # 数据迭代器

    frame = Contrastive_Frame(augment=augment, 
                           model_name=f'{encoder_model_name}_{config_model_name}', 
                           epochs=epochs+warm_up_epochs, 
                           min_lr=min_lr, 
                           device=device, 
                           if_full_cpu=if_full_cpu,
                           feature_map_layer_n=FEATURE_MAP_LAYER_NAMES,
                           feature_map_num=FEATURE_MAP_NUM,
                           feature_map_interval=FEATURE_MAP_INTERVAL,
                           use_data_parallel=USE_DATA_PARALLEL,
                           display_nums=DISPLAY_NUMS)
    
    train(frame=frame,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          dataloader=dataloader,
          ck_pth=ck_pth, )