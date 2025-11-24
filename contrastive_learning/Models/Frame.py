import sys, os
import shutil
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import cpu_count
import torch
from utils import AverageMeter, ProgressMeter, topk_accuracy
import numpy as np
from torch.nn import DataParallel
import re
import logging

try: # 使用swanlab进行实验管理
    import swanlab
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

class Contrastive_Frame:
    def __init__(self, augment, model_name, min_lr=1e-7, epochs=300, device=None, if_full_cpu=True
                 ,feature_map_layer_n=None, feature_map_num=12, feature_map_interval=10,
                 use_data_parallel=False, display_nums=10):
        self.augment = augment
        self.loss = nn.CrossEntropyLoss()
        self.min_lr = min_lr
        self.epochs=epochs
        self.use_data_parallel = use_data_parallel # 是否使用DataParallel进行多GPU训练
        self.display_nums = display_nums
        self.model_name = model_name
        self.logger = None

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else: self.device = device

        #配置训练信息
        self.if_full_cpu = if_full_cpu
        self.train_epoch_min_loss = 100
        self.epoch_max_acc = -1
        self.start_epoch = 0

        self.feature_map_layer_n = feature_map_layer_n
        self.feature_map_num = feature_map_num
        self.feature_map_interval = feature_map_interval

    def init_files(self, id=None): # 初始化文件夹和日志文件, id用于swanlab断点续训
        current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
        if id is not None:
            model_save_name = f'{self.model_name}_{current_time}_ID{id}'
        else:
            model_save_name = f'{self.model_name}_{current_time}'
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.parent_dir = os.path.join(base_dir, '_results')  # 创建一个父目录保存训练结果
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)
        self.model_dir = os.path.join(self.parent_dir, model_save_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # 配置输出模型的名称和日志名称
        self.model_path = os.path.join(self.model_dir, f'{model_save_name}.pth')
        self.model_best_path = os.path.join(self.model_dir, f'{model_save_name}_best.pth')
        self.model_best_path_pt = os.path.join(self.model_dir, f'{model_save_name}_best.pt')
        self.log_path = os.path.join(self.model_dir, f'{model_save_name}.log')
        self._setup_logger()
    
    def _setup_logger(self):
        """配置 logging"""
        self.logger = logging.getLogger(self.model_name)
        self.logger.propagate = False # 防止日志重复打印
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = [] # 清除已有的 handler，防止重复打印

        # File Handler (写入文件)
        file_handler = logging.FileHandler(self.log_path, mode='w', encoding='utf-8')
        file_formatter = logging.Formatter('%(message)s') # 文件只记录消息内容
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console Handler (输出到控制台)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s') # 控制台只显示消息
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def find_id(self, ck_pth): # 从ckpt路径中提取ID
        if ck_pth is None:
            return None
        ck_pth = os.path.basename(ck_pth)
        pattern = r"_ID([^\.]+)\."
        match = re.search(pattern, ck_pth)
        if match:
            return match.group(1)[:21]
        return None

    def full_cpu(self):
        cpu_num = cpu_count()  # 自动获取最大核心数目
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        if self.if_full_cpu:
            torch.set_num_threads(cpu_num)
            print('Using cpu core num: ', cpu_num)

    def check_input(self, model):
        """检查模型与所有参数的兼容性"""
        if SWANLAB_AVAILABLE:
            if self.feature_map_layer_n is not None: # 检查指定的绘制特征图的层名是否存在于模型中
                for name in self.feature_map_layer_n:
                    if name not in dict(model.encoder_q.named_modules()).keys():
                        raise ValueError(f"Layer name '{name}' not found in the model's modules. \
                                         The available layers are:\n {list(model.encoder_q.named_modules().keys())}")

    def _reshape_transform_3d(self, tensor):    
        # 输入可能是 [N, C, D, H, W]，对 D 维做平均，返回 [N, C, H, W]
        if tensor.ndim == 5:
            return tensor.mean(dim=2)
        return tensor

    def draw_feature_maps(self, model, data_list, epoch):
        """绘制并保存特征图, model必须是encoder_q"""
        train_mode = model.training
        if isinstance(model, DataParallel):
            model = model.module.encoder_q
        else:
            model = model.encoder_q
        model.eval()
        data = data_list[0]
        max_batch_features = min(data.size(0), self.feature_map_num)
        data = data[:max_batch_features] # 仅使用前N个图像绘制特征图
        module_dict = dict(model.named_modules())
        if not self.feature_map_layer_n: # 如果目标层为空
            feature_map_layer_n = []
            for name, module in module_dict.items(): # 返回所有模块的名称和模块本身
                if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                    feature_map_layer_n.append(name)
            if len(feature_map_layer_n) > 2: # 如果层数过多，则只取前三层、中间层、最后一层
                self.feature_map_layer_n = [feature_map_layer_n[0], feature_map_layer_n[len(feature_map_layer_n) // 2], feature_map_layer_n[-1]]
            else:
                self.feature_map_layer_n = [] 

        for layer_name in self.feature_map_layer_n: # 对每个指定层绘制GradCAM
            target_layer = [module_dict[layer_name]]
            cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=self._reshape_transform_3d)
            grayscale_cam = cam(input_tensor=data) # targets默认为None，自动指定模型预测概率最大的类
            visualization_list = []
            for i in range(max_batch_features):
                input_img = data[i].cpu().numpy().transpose(1,2,0)  # 转换为 (H,W,C)
                input_img = stretch_img(input_img[:,:,:3])  # 拉伸到0-1范围
                visualization = show_cam_on_image(input_img, grayscale_cam[i], use_rgb=True)
                visualization_list.append(visualization)
            swanlab.log({f"GradCAMgray/{layer_name}": [swanlab.Image((grayscale_cam[i] * 255).astype(np.uint8))
                                                       for i in range(max_batch_features)]}, step=epoch) # 绘制前N个图像的GradCAM灰度图
            swanlab.log({f"GradCAMoverlay/{layer_name}": [swanlab.Image(visualization_list[i])
                                                          for i in range(max_batch_features)]}, step=epoch)
        if train_mode:
            model.train()

def stretch_img(arr):
    """将输入的numpy数组按通道拉伸为0-1的np.float32格式
    arr: np.ndarray, (H,W) or (H, W, 3)"""
    arr = arr.astype(np.float32)
    if arr.ndim == 3:  # (C,H,W)
        min_val = arr.min(axis=(0, 1), keepdims=True)
        max_val = arr.max(axis=(0, 1), keepdims=True)
        stretched = (arr - min_val) / (max_val - min_val + 1e-8)
    elif arr.ndim == 2:  # (H,W)
        min_val = arr.min()
        max_val = arr.max()
        stretched = (arr - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return stretched

def save_model(frame, model, optimizer, scheduler, epoch=None, avg_loss=None, avg_acc=None, is_best=False):
    """注意：将需要迁移的部分使用 backbone 存储起来"""
    if isinstance(model, DataParallel):
        model = model.module  # 解包DataParallel
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': avg_loss,
        'best_acc': avg_acc,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'current_lr': optimizer.param_groups[0]['lr'],
        'backbone': model.encoder_q.encoder.state_dict(),
    }
    torch.save(state, frame.model_path)
    if is_best:
        shutil.copyfile(frame.model_path, frame.model_best_path)
        torch.save(model.encoder_q, frame.model_best_path_pt) # 保存整个模型结构
        print(f"============The best checkpoint saved at epoch {epoch}============")

def get_leaf_layers_info(model):
    """
    获取模型的叶子层（底层模块）信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        dict: 叶子层名称和描述的字典
    """
    leaf_layers = []
    for name, module in model.named_modules():
        if name == '':  # 跳过根模块
            continue
        if len(list(module.children())) == 0:
            module_desc = str(module)
            leaf_layers.append(f"({name}) ({module_desc})")
    return leaf_layers


def load_parameter(frame, model, optimizer, scheduler=None, ck_pth=None): # 加载模型、优化器、调度器
    frame.full_cpu() # 打印配置信息
    if ck_pth is not None:
        checkpoint = torch.load(ck_pth, weights_only=True, map_location=frame.device)  # 加载断点
        model.load_state_dict(checkpoint['model'])
        frame.train_epoch_min_loss = checkpoint.get('best_loss', 100)
        try:
            optimizer.load_state_dict(checkpoint['optimizer']) # 恢复优化器
            print('The optimizer state have been loaded!')
            frame.start_epoch = checkpoint.get('epoch', -1)
        except(ValueError, RuntimeError):
            print('The optimizer is incompatible, and the parameters do not match')
        if scheduler and 'scheduler' in checkpoint: # 恢复调度器
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except (ValueError, RuntimeError):
                print('The scheduler is incompatible')
        print(f"Loaded checkpoint from epoch {frame.start_epoch}, current lr {optimizer.param_groups[0]['lr']}")

def train(frame, model, optimizer, dataloader, scheduler=None, ck_pth=None):
    ID = frame.find_id(ck_pth) # 从ckpt路径中提取swanlab的ID
    frame.check_input(model)
    IF_DRAW_FEATURE_MAPS = model.encoder_q.if_draw_feature_maps
    model.to(frame.device)
    load_parameter(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, ck_pth=ck_pth) # 初始化模型
    if SWANLAB_AVAILABLE:
        run = swanlab.init(
            project="Contrastive_Learning",
            experiment_name=f"{type(model).__name__}_{type(model.encoder_q.encoder).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_name": type(model.encoder_q.encoder).__name__,
                "init_learning_rate": optimizer.param_groups[0]['lr'],
                "min_learning_rate": frame.min_lr,
                "epochs": frame.epochs,
                "batch_size": dataloader.batch_size,
                "single-epoch_dataset_size": len(dataloader.dataset),
                "device": str(frame.device),
                "use_data_parallel": frame.use_data_parallel,
                "module": get_leaf_layers_info(model)
            },
            resume=True if ck_pth is not None else False, # 控制断点续训
            id=ID,
        )
        frame.init_files(id=run.id)  # 初始化文件夹和日志文件
    else:
        frame.init_files(id=None)
    if frame.use_data_parallel:
        print(f"Use DataParallel, The number of GPUs is: {torch.cuda.device_count()}")
        model = DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        print(f'Cuda device count: {torch.cuda.device_count()} And the current device:{frame.device}')  # 显卡数
    model.to(frame.device)
    
    model_save_epoch = 0
    max_iter_num = len(dataloader)
    display_nums = frame.display_nums if frame.display_nums > 1 else 0.1
    interval_printinfo = int(max_iter_num // display_nums)
    loss_note = AverageMeter("Loss", ":.6f")
    top1_acc_note = AverageMeter("Top1-Accuracy", ":.4f")
    top5_acc_note = AverageMeter("Top5-Accuracy", ":.4f") 
    Epoch_wirter = ProgressMeter(frame.epochs, len(dataloader), [loss_note, top1_acc_note, top5_acc_note], "\nStep")

    DAMPLES = []
    # start training
    for epoch in range(frame.start_epoch+1, frame.epochs+1):
        model.train()
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'\n{formatted_time} Epoch {epoch}:')
        for i,block in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train", leave=True):
            block = block.to(frame.device) # batch,C,H,W
            batchs = block.size(0)
            with torch.no_grad():
                q = frame.augment(block)
                k = frame.augment(block)
            if not DAMPLES and SWANLAB_AVAILABLE: # 仅在第一个epoch保存样本用于绘图
                max_draw_samples = min(batchs, frame.feature_map_num)
                DAMPLES.append(block[:max_draw_samples].detach().cpu())
                swanlab.log(
                {
                    "samples/original": [swanlab.Image(img[:3]) for img in DAMPLES[0][:max_draw_samples]],
                    "samples/augmented_q": [swanlab.Image(img[:3]) for img in q.detach()[:max_draw_samples]],
                    "samples/augmented_k": [swanlab.Image(img[:3]) for img in k.detach()[:max_draw_samples]],
                })
            optimizer.zero_grad()  # 清空梯度
            if isinstance(model, DataParallel):
                model.module.update_key_encoder() # 更新key encoder
                q, k = model(q, k) # 前向计算
                logits, label = model.module.calculate_logits(q, k) # 计算logits和labels
            else:
                model.update_key_encoder() # 更新key encoder
                q, k = model(q, k) # 前向计算
                logits, label = model.calculate_logits(q, k) # 计算logits和labels
            loss = frame.loss(logits, label)
            acc1, acc5 = topk_accuracy(logits, label, topk=(1, 5))

            loss_note.update(loss.item(), batchs)
            top1_acc_note.update(acc1.item(), batchs)
            top5_acc_note.update(acc5.item(), batchs)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            if (i+1) % interval_printinfo == 0:
                Epoch_wirter.display(i+1)

        DRAW_FEATURE_MAPS = True if (epoch % frame.feature_map_interval == 0 or epoch == 1) and SWANLAB_AVAILABLE \
            and IF_DRAW_FEATURE_MAPS else False
        if DRAW_FEATURE_MAPS:
            frame.draw_feature_maps(model, DAMPLES, epoch)
        
        dataloader.dataset.reset()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_summary = Epoch_wirter.epoch_summary(epoch, f"Lr:{current_lr:.2e}")
        if SWANLAB_AVAILABLE:
            swanlab.log(
                {
                    "Loss": loss_note.avg,
                    "Top1_Acc": top1_acc_note.avg,
                    "Top5_Acc": top5_acc_note.avg,
                    "Learning_rate": current_lr
                },
                step=epoch
            )
        frame.logger.info(epoch_summary)
        is_best = False
        if top1_acc_note.avg > frame.epoch_max_acc: # 使用top1准确率来保存模型
            frame.epoch_max_acc = top1_acc_note.avg
            frame.train_epoch_min_loss = loss_note.avg
            model_save_epoch = epoch
            is_best = True
        elif top1_acc_note.avg == frame.epoch_max_acc:
            if loss_note.avg < frame.train_epoch_min_loss:
                frame.epoch_max_acc = top1_acc_note.avg
                frame.train_epoch_min_loss = loss_note.avg
                model_save_epoch = epoch
                is_best = True
        save_model(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, 
                    epoch=epoch, avg_loss=loss_note.avg, avg_acc=top1_acc_note.avg, is_best=is_best)
        if scheduler is not None:
            scheduler.step()
        loss_note.reset()
        top1_acc_note.reset()
        top5_acc_note.reset()
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = f'{formatted_time} Model saved at Epoch{model_save_epoch}. The best top1-acc:{frame.epoch_max_acc}. The best training_loss:{frame.train_epoch_min_loss}'
    frame.logger.info(result)
  
class Contrasive_learning_predict_frame:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device
        self.out_embedding = None
    
    def predict(self, model, dataloader):
        model.to(self.device)
        with torch.no_grad():
            model.eval()
            idx = 0
            for image in tqdm(dataloader, total=len(dataloader)):
                image = image.to(self.device)
                predict = model(image)
                if self.out_embedding is None:
                    # 初始化输出嵌入矩阵，预分配内存
                    embedding_nums = predict.shape[-1]
                    self.out_embedding = torch.empty((len(dataloader.dataset), embedding_nums), dtype=torch.float32, device=self.device)
                self.out_embedding[idx:idx+len(predict)] = predict
                idx += len(predict)
        self.out_embedding = self.out_embedding.cpu().numpy()
        return self.out_embedding
