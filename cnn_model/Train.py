"""针对Train模块的升级版"""
import sys
sys.path.append('.')
import torch
import torch.optim as optim
from cnn_model.Models.Scheduler import WarmupLinearSchedule
from cnn_model.Models.Models import SRACN, Common_1DCNN, Common_3DCNN, SSRN, HybridSN, Vgg16_net, MobileNetV1, \
                                    MobileNetV2, Common_2DCNN, ResNet18, Res_3D_18Net, Res_3D_34Net, Res_3D_50Net, \
                                    ResNet34, ResNet50, spec_transformer
from cnn_model.Models.Data import CNN_Dataset
from cnn_model.Models.Frame import Cnn_Model_Frame, train
from utils import read_dataset_from_txt
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import random
MODEL_DICT = {
    'SRACN':SRACN,
    'Common_1DCNN': Common_1DCNN,
    'Common_2DCNN': Common_2DCNN,
    'Common_3DCNN': Common_3DCNN,
    "Res_3D_18Net": Res_3D_18Net,
    "Res_3D_34Net": Res_3D_34Net,
    "Res_3D_50Net": Res_3D_50Net,
    'SSRN': SSRN,
    'HybridSN': HybridSN,
    'Vgg16': Vgg16_net,
    'MobileNetV1': MobileNetV1,
    'MobileNetV2': MobileNetV2,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'spec_transformer': spec_transformer,
}

model_selected = 'spec_transformer' # 从上面选择一个模型
config_name = "Test" # 配置输出名称，最后的输出名称为 model_selected_config_name_CurrentTime
train_images_dir = r'c:\Users\85002\Desktop\test\test\train_dataset\.datasets.txt'  # 训练数据集
test_images_dir = r'c:\Users\85002\Desktop\test\test\test_dataset\.datasets.txt'  # 测试数据集
out_classes = 11 # 分类数


epochs = 100 # epoch
batch = 48 # batch
init_lr = 3e-4  # lr
min_lr = 3e-5  # 最低学习率
warm_up_epochs = 20  # 预热epoch数
pretrain_pth = None
ck_pth = None # 用于断点学习
if_full_cpu = True  # 是否全负荷cpu
USE_DATA_PARALLEL = False # 是否使用DataParallel进行多显卡训练

"""特征图绘制相关参数"""
FEATURE_MAP_LAYER_NAMES = [] # 指定需要绘制特征图的层名，使用列表形式，例如 ['encoder','layer1.0.conv1']，如果为空
FEATURE_MAP_NUM = 36 # 每个层绘制的特征图数量
FEATURE_MAP_POSITION = 0.2 # 在测试集中的位置，范围0-1之间，例如0.5表示在测试集的中间位置绘制特征图(不能精确控制具体位置，只能大致控制)
FEATURE_MAP_INTERVAL = 10 # 每隔多少个epoch绘制一次特征图
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 显卡设置
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        USE_DATA_PARALLEL = False

    dataloader_num_workers = cpu_count() // 4 # 根据cpu核心数自动决定num_workers数量
    print(f"🔍 PyTorch Version: {torch.__version__}")
    print(f'📊 Using num_workers: {dataloader_num_workers}')
    # 配置训练数据集和模型
    train_image_lists = read_dataset_from_txt(train_images_dir)
    test_image_lists = read_dataset_from_txt(test_images_dir)
    list_shuffler = random.Random(42)
    list_shuffler.shuffle(test_image_lists)
    train_dataset = CNN_Dataset(train_image_lists)
    eval_dataset = CNN_Dataset(test_image_lists)
    model = MODEL_DICT[model_selected](out_classes=out_classes, in_shape=train_dataset.data_shape)  # 模型实例化
    print(f"🎯 Image shape: {train_dataset.data_shape}")
    if pretrain_pth is not None:
        state_dict = torch.load(pretrain_pth, map_location=device)["backbone"]
        model._load_encoer_params(state_dict) # 加载预训练权重
        model._freeze_encoder() # 冻结编码器参数
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-4)  # 优化器
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warm_up_epochs, t_total=epochs+warm_up_epochs, min_lr=min_lr)  # 学习率调度器
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, pin_memory=True, 
                                  num_workers=dataloader_num_workers, prefetch_factor=2,
                                  persistent_workers=dataloader_num_workers > 0)  # 数据迭代器
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch, shuffle=False, pin_memory=True, 
                                 num_workers=dataloader_num_workers, prefetch_factor=2,
                                 persistent_workers=dataloader_num_workers > 0)  # 数据迭代器

    frame = Cnn_Model_Frame(model_name=f'{model_selected}_{config_name}', 
                            epochs=epochs+warm_up_epochs, 
                            min_lr=min_lr,
                            device=device, 
                            if_full_cpu=if_full_cpu,
                            feature_map_layer_n=FEATURE_MAP_LAYER_NAMES,
                            feature_map_num=FEATURE_MAP_NUM,
                            feature_map_position=FEATURE_MAP_POSITION,
                            feature_map_interval=FEATURE_MAP_INTERVAL,
                            use_data_parallel=USE_DATA_PARALLEL)
    
    train(frame=frame,
          model=model, 
          optimizer=optimizer, 
          scheduler=scheduler,
          train_dataloader=train_dataloader, 
          eval_dataloader=eval_dataloader,
          ck_pth=ck_pth)