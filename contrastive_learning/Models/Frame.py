import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import shutil
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import cpu_count
import torch
import traceback
from utils import AverageMeter, ProgressMeter, topk_accuracy

try: # 使用swanlab进行试验管理
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

class Contrastive_Frame:
    def __init__(self, augment, model_name, min_lr=1e-7, epochs=300, device=None, if_full_cpu=True
                 ,feature_map_layer_n=None, feature_map_num=12):
        self.augment = augment
        self.loss = nn.CrossEntropyLoss()
        self.min_lr = min_lr
        self.epochs=epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device

        # 配置输出模型的名称和日志名称
        current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
        model_save_name = f'{model_name}_{current_time}'
        self.parent_dir = os.path.join(base_path, '_results') # 创建一个父目录保存训练结果
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)
        self.model_dir = os.path.join(self.parent_dir, model_save_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.model_path = os.path.join(self.model_dir, f'{model_save_name}.pth')
        self.model_best_path = os.path.join(self.model_dir, f'{model_save_name}_best.pth')
        self.model_best_path_pt = os.path.join(self.model_dir, f'{model_save_name}_best_encoder.pt')
        self.log_path = os.path.join(self.model_dir, f'{model_save_name}.log')

        #配置训练信息
        self.if_full_cpu = if_full_cpu
        self.train_epoch_min_loss = 100
        self.epoch_max_acc = -1
        self.start_epoch = 0

        self.feature_maps = {}  # 存储特征图
        self.hook_handles = []  # 存储hook句柄
        self.feature_map_layer_n = feature_map_layer_n
        self.feature_map_num = feature_map_num

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
        print(f'Cuda device count: {torch.cuda.device_count()} And the current device:{self.device}')  # 显卡数

    def check_input(self, model):
        """检查模型与所有参数的兼容性"""
        if SWANLAB_AVAILABLE:
            if self.feature_map_layer_n is not None: # 检查指定的绘制特征图的层名是否存在于模型中
                for name in self.feature_map_layer_n:
                    if name not in dict(model.named_modules()).keys():
                        raise ValueError(f"Layer name '{name}' not found in the model's modules. The available layers are:\n {list(model.named_modules().keys())}")

    def _hook_fn(self, module, input, output, layer_name=None):
        """Hook函数，用于捕获特征图"""
        self.feature_maps[layer_name] = output.detach().cpu()  # 将特征图存储到字典中

    def register_hooks(self, model):
        """注册hook来捕获特征图, 不指定层名, 则默认注册第一个、中间层与最后一个卷积层"""
        # 如果未指定层名，自动获取卷积层
        if self.feature_map_layer_n is None:
            self.feature_map_layer_n = []
            q_feature_map_layer_n = []
            k_feature_map_layer_n = []
            for name, module in model.named_modules(): # 返回所有模块的名称和模块本身
                if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                    if 'encoder_q' in name:
                        q_feature_map_layer_n.append(name)
                    else:
                        k_feature_map_layer_n.append(name)
            q_feature_map_layer_n = [q_feature_map_layer_n[0], q_feature_map_layer_n[len(q_feature_map_layer_n) // 2], 
                                    q_feature_map_layer_n[-1]]  # 仅注册第一个、中间层与最后一个卷积层
            k_feature_map_layer_n = [k_feature_map_layer_n[0], k_feature_map_layer_n[len(k_feature_map_layer_n) // 2], 
                                    k_feature_map_layer_n[-1]]  # 仅注册第一个、中间层与最后一个卷积层
            self.feature_map_layer_n = q_feature_map_layer_n + k_feature_map_layer_n

        # 注册hook
        for name, module in model.named_modules():
            if name in self.feature_map_layer_n:
                self.hook_handles.append(module.register_forward_hook(
                    lambda m, i, o, layer_name=name: self._hook_fn(m, i, o, layer_name=layer_name))) # 保存hook句柄以便后续移除
                
    def remove_hooks(self, epoch):
        """移除所有注册的hook与特征图"""
        if self.hook_handles:
            self._draw_feature_maps(epoch)  # 移除前先绘制特征图
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []
            self.feature_maps.clear()
    
    def _draw_feature_maps(self, epoch):
        """绘制并保存特征图"""
        if self.feature_maps:
            for layer_name, feature_map in self.feature_maps.items():
                encoder = 'encoder_q' if 'encoder_q' in layer_name else 'encoder_k'
                name = layer_name[9:]
                ndim = feature_map.ndim
                if ndim not in [4, 5]:
                    pass
                elif ndim == 4: # 2D卷积特征图, 同时绘制单通道图（单样本）与前三个通道的假彩色图（多样本）
                    max_channel_features = min(feature_map.size(1), self.feature_map_num)
                    max_batch_features = min(feature_map.size(0), self.feature_map_num)
                    swanlab.log({f"feature_maps_{encoder}/{name}_single_band": [swanlab.Image(img) for img in feature_map[0, :max_channel_features]]}, step=epoch)
                    swanlab.log({f"feature_maps_{encoder}/{name}_band_rgb": [swanlab.Image(img[:3]) for img in feature_map[:max_batch_features]]}, step=epoch)
                elif ndim == 5: # 3D卷积特征图, 同时绘制单通道图（单样本）与前三个通道的假彩色图（多样本）
                    max_channel_features = min(feature_map.size(1), self.feature_map_num)
                    max_batch_features = min(feature_map.size(0), self.feature_map_num)
                    swanlab.log({f"feature_maps_{encoder}/{name}_single_channel": [swanlab.Image(img[:3]) for img in feature_map[0, :max_channel_features]]}, step=epoch)
                    swanlab.log({f"feature_maps_{encoder}/{name}_single_channel_band_rgb": [swanlab.Image(img[0, :3]) for img in feature_map[:max_batch_features]]}, step=epoch)
            else:
                print(f"Feature map dimension: {ndim} not supported for visualization.")
        else:
            print("No feature maps to draw.")

def save_model(frame, model, optimizer, scheduler, epoch=None, avg_loss=None, avg_acc=None, is_best=False):
    """注意：将需要迁移的部分使用 backbone 存储起来"""
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

def clean_up(frame):
    """清理因终端或异常生成的文件"""
    if not os.path.exists(frame.model_path):
        if os.path.exists(frame.model_dir):
            shutil.rmtree(frame.model_dir)
            print(f"Directory {frame.model_dir} has been removed.")
    else: pass


def load_parameter(frame, model, optimizer, scheduler=None, ck_pth=None): # 加载模型、优化器、调度器
    frame.full_cpu() # 打印配置信息
    if ck_pth is not None:
        checkpoint = torch.load(ck_pth, weights_only=True, map_location=frame.device)  # 加载断点
        model.load_state_dict(checkpoint['model'])
        frame.train_epoch_min_loss = checkpoint.get('best_loss', 100)
        try:
            optimizer.load_state_dict(checkpoint['optimizer']) # 恢复优化器
            print('The optimizer state have been loaded!')
            frame.start_epoch = checkpoint.get('epoch', -1) + 1  # 获取epoch信息，如果没有，默认为0
        except(ValueError, RuntimeError):
            print('The optimizer is incompatible, and the parameters do not match')
        if scheduler and 'scheduler' in checkpoint: # 恢复调度器
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except (ValueError, RuntimeError):
                print('The scheduler is incompatible')
        print(f"Loaded checkpoint from epoch {frame.start_epoch}, current lr {optimizer.param_groups[0]['lr']}")

def train(frame, model, optimizer, dataloader, scheduler=None, ck_pth=None):
    def finish_work():
        try:
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = f'{formatted_time} Model saved at Epoch{model_save_epoch}. The best top1-acc:{frame.epoch_max_acc}. The best training_loss:{frame.train_epoch_min_loss}'
            log_writer.write(result + '\n')
            if log_writer is not None: # 确保日志文件被正确关闭
                log_writer.close()
        except:pass
    
    if SWANLAB_AVAILABLE:
        swanlab.init(
            project="Contrastive_Learning",
            experiment_name=f"{type(model).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_name": type(model.encoder_q.encoder).__name__,
                "init_learning_rate": optimizer.param_groups[0]['lr'],
                "min_learning_rate": frame.min_lr,
                "epochs": frame.epochs,
                "batch_size": dataloader.batch_size,
                "single-epoch_dataset_size": len(dataloader.dataset),
                "device": str(frame.device),
                "module": get_leaf_layers_info(model)
            }
        )
    log_writer = open(frame.log_path, 'w')
    model.to(frame.device)
    load_parameter(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, ck_pth=ck_pth) # 初始化模型
    
    model_save_epoch = 0
    max_iter_num = len(dataloader)
    interval_printinfo = max_iter_num // 10 # 每个epoch打印十次过程参数
    loss_note = AverageMeter("Loss", ":.6f")
    top1_acc_note = AverageMeter("Top1-Accuracy", ":.4f")
    top5_acc_note = AverageMeter("Top5-Accuracy", ":.4f") 
    Epoch_wirter = ProgressMeter(frame.epochs, len(dataloader), [loss_note, top1_acc_note, top5_acc_note], "\nStep")

    samples_draw = []
    # start training
    try:
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
                if not samples_draw and SWANLAB_AVAILABLE: # 仅在第一个epoch保存样本用于绘图
                    samples_draw.append(block.detach().cpu())
                    samples_draw.append(q.detach().cpu())
                    samples_draw.append(k.detach().cpu())
                    max_draw_samples = min(batchs, frame.feature_map_num)
                    swanlab.log( # 绘制样本图像
                    {
                        "samples/original": [swanlab.Image(img[:3]) for img in samples_draw[0][:max_draw_samples]],
                        "samples/augmented_q": [swanlab.Image(img[:3]) for img in samples_draw[1][:max_draw_samples]],
                        "samples/augmented_k": [swanlab.Image(img[:3]) for img in samples_draw[2][:max_draw_samples]],
                    })
                optimizer.zero_grad()  # 清空梯度
                logits, label = model(q, k)
                loss = frame.loss(logits, label)
                acc1, acc5 = topk_accuracy(logits, label, topk=(1, 5))

                loss_note.update(loss.item(), batchs)
                top1_acc_note.update(acc1.item(), batchs)
                top5_acc_note.update(acc5.item(), batchs)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                if (i+1) % interval_printinfo == 0:
                    Epoch_wirter.display(i+1)

            DRAW_FEATURE_MAPS = True if (epoch % 10 == 0 or epoch == 1) and SWANLAB_AVAILABLE else False # 每10个epoch绘制一次特征图
            if DRAW_FEATURE_MAPS:
                frame.register_hooks(model)
                with torch.no_grad():
                    model.eval()
                    _ = model(samples_draw[1].to(frame.device), samples_draw[2].to(frame.device))  # 前向传播以捕获特征图
                frame.remove_hooks(epoch) # 绘制特征图并移除hooks
            
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
            log_writer.write(epoch_summary + '\n') # 记录训练过程
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
            if current_lr <= frame.min_lr:
                pass
            else:
                if scheduler is not None:
                    scheduler.step()
            loss_note.reset()
            top1_acc_note.reset()
            top5_acc_note.reset()
            log_writer.flush()
        finish_work()
    except KeyboardInterrupt: # 捕获键盘中断信号
        finish_work()
        print(f"Training interrupted due to: KeyboardInterrupt")
        clean_up(frame=frame)
    except Exception as e: 
        print(traceback.format_exc())  # 打印完整的堆栈跟踪
        clean_up(frame=frame)
    finally:
        print(f"Training completed. Program exited.")
        sys.exit(0)
  
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
