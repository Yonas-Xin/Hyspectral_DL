import os, sys
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm
from multiprocessing import cpu_count
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import torch
import shutil
from utils import AverageMeter, ProgressMeter, topk_accuracy
import traceback

try: # 使用swanlab进行试验管理
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

class Cnn_Model_Frame:
    def __init__(self, model_name, min_lr=1e-7, epochs=300, device=None, if_full_cpu=True, 
                 feature_map_layer_n=[], feature_map_num=12, feature_map_position=0,):
        self.loss_func = nn.CrossEntropyLoss()
        self.min_lr = min_lr
        self.epochs = epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device

        # 配置输出模型的名称和日志名称
        current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
        model_save_name = f'{model_name}_{current_time}'
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.parent_dir = os.path.join(base_dir, '_results')  # 创建一个父目录保存训练结果
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)
        self.model_dir = os.path.join(self.parent_dir, model_save_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.model_path = os.path.join(self.model_dir, f'{model_save_name}.pth')
        self.model_best_path = os.path.join(self.model_dir, f'{model_save_name}_best.pth')
        self.model_best_path_pt = os.path.join(self.model_dir, f'{model_save_name}_best.pt')
        self.log_path = os.path.join(self.model_dir, f'{model_save_name}.log')

        # 配置训练信息、用于断点训练
        self.if_full_cpu = if_full_cpu
        self.test_epoch_min_loss = 100
        self.test_epoch_max_acc = -1
        self.start_epoch = 0

        # 存储最佳模型的预测结果
        self.best_all_labels = None
        self.best_all_preds = None

        self.feature_maps = {}  # 存储特征图
        self.hook_handles = []  # 存储hook句柄
        self.feature_map_layer_n = feature_map_layer_n
        self.feature_map_num = feature_map_num
        self.feature_map_position = feature_map_position

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
    def _hook_fn_input(self, module, input, output):
        """Hook函数，用于捕获输入的图像"""
        self.feature_maps["input"] = input[0].detach().cpu()  # 将特征图存储到字典中,注意,input是一个元组
    def _register_input_hook(self, model):  # 注册hook来捕获输入图像
        """注册hook来捕获输入图像"""
        first_module = list(model.named_children())[0][1]  # 获取模型的第一个模块, 如果第一个模块注册了hook, hook会注册顺序调用
        handle = first_module.register_forward_hook(self._hook_fn_input)
        self.hook_handles.append(handle)
    
    def register_hooks(self, model):
        """注册hook来捕获特征图, 不指定层名, 则默认注册第一个、中间层与最后一个卷积层"""
        # 如果未指定层名，自动获取卷积层
        if not self.feature_map_layer_n: # 如果层名为空
            for name, module in model.named_modules(): # 返回所有模块的名称和模块本身
                if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                    self.feature_map_layer_n.append(name)
            self.feature_map_layer_n = [self.feature_map_layer_n[0], self.feature_map_layer_n[len(self.feature_map_layer_n) // 2], 
                                        self.feature_map_layer_n[-1]]  # 仅注册第一个、中间层与最后一个卷积层
        # 注册hook
        # self._register_input_hook(model)
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
            self.hook_handles.clear()
            self.feature_maps.clear()
    
    def _draw_feature_maps(self, epoch):
        """绘制并保存特征图"""
        if self.feature_maps:
            for layer_name, feature_map in self.feature_maps.items():
                ndim = feature_map.ndim
                if ndim not in [4, 5]:
                    pass
                elif ndim == 4:
                    max_channel_features = min(feature_map.size(1), self.feature_map_num)
                    max_batch_features = min(feature_map.size(0), self.feature_map_num)
                    swanlab.log({f"feature_maps/{layer_name}_single_channel_maps": [swanlab.Image(img) for img in feature_map[0, :max_channel_features]]}, step=epoch) # 绘制第一个图像的前12个通道图
                    swanlab.log({f"feature_maps/{layer_name}_batch_rgb_maps": [swanlab.Image(pca_torch(img)) for img in feature_map[:max_batch_features]]}, step=epoch) # 绘制前12个图像的前三个通道假彩色图
                elif ndim == 5:
                    max_channel_features = min(feature_map.size(1), self.feature_map_num)
                    max_batch_features = min(feature_map.size(0), self.feature_map_num)
                    swanlab.log({f"feature_maps/{layer_name}_single_channel_maps": [swanlab.Image(img[:3]) for img in feature_map[0, :max_channel_features]]}, step=epoch) # 绘制第一个图像的前12个通道的前三个波段假彩色图
                    swanlab.log({f"feature_maps/{layer_name}_batch_rgb_maps": [swanlab.Image(img[0, :3]) for img in feature_map[:max_batch_features]]}, step=epoch) # 绘制前12个图像的第0个通道的前三个波段假彩色图
                else:
                    print(f"Feature map dimension: {ndim} not supported for visualization.")
        else:
            print("No feature maps to draw.")

def pca_torch(tensor):
    """
    使用PyTorch的SVD实现PCA，将多通道(c,h,w)张量降维为(3,h,w)
    这个版本更高效，不需要在numpy和torch之间转换
    """
    if len(tensor.shape) != 3:
        raise ValueError(f"输入张量应该是3维的(c,h,w)，但得到的是{tensor.shape}")
    c, h, w = tensor.shape
    x = tensor.view(c, -1)  # (c, h*w)
    x_centered = x - x.mean(dim=1, keepdim=True)
    U, S, V = torch.svd(x_centered.T)  # V的形状是(c, c)
    components = V[:, :3]  # (c, 3)
    reduced = x_centered.T @ components  # (h*w, 3)
    reduced_tensor = reduced.T.view(3, h, w)
    return reduced_tensor

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
        checkpoint = torch.load(ck_pth, weights_only=True, map_location=frame.device)
        model.load_state_dict(checkpoint['model'])
        frame.test_epoch_min_loss = checkpoint.get('best_loss', 100)
        frame.test_epoch_max_acc = checkpoint.get('best_acc', -1)
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

def save_model(frame, model, optimizer, scheduler, epoch=None, avg_loss=None, avg_acc=None, is_best = False):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': avg_loss,
        'best_acc': avg_acc,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'current_lr': optimizer.param_groups[0]['lr']
    }
    torch.save(state, frame.model_path)
    if is_best:
        shutil.copyfile(frame.model_path, frame.model_best_path)
        torch.save(model, frame.model_best_path_pt) # 保存整个模型结构
        print(f"============The best checkpoint saved at epoch {epoch}============")

def train(frame, model, optimizer, train_dataloader, eval_dataloader=None, scheduler=None, ck_pth=None):
    def finish_work(): # 结束工作，打印最终结果
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best_result = f'{formatted_time} Model saved at Epoch {model_save_epoch}. The best training_acc:{best_train_accuracy:.4f}%. The best testing_acc:{best_test_accuracy:.4f}%.'
        if SWANLAB_AVAILABLE:
            pass
            # matrics = swanlab.confusion_matrix(frame.best_all_labels, frame.best_all_preds, class_names=None) # 生成混淆矩阵,这个函数有问题，等修复
            # swanlab.log({"confusion_matrix_custom": matrics})
            # log_writer = None
        log_writer.write(best_result + '\n')
        print(best_result)
        print_report(frame.best_all_labels.tolist(), frame.best_all_preds.tolist(), log_writer=log_writer)
        if log_writer is not None: # 确保日志文件被正确关闭
            log_writer.close()
    
    frame.check_input(model) # 检查模型与参数的兼容性
    if SWANLAB_AVAILABLE:
        swanlab.init(
            project="Cnn_Model_Training",
            experiment_name=f"{type(model).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_name": type(model).__name__,
                "init_learning_rate": optimizer.param_groups[0]['lr'],
                "min_learning_rate": frame.min_lr,
                "epochs": frame.epochs,
                "batch_size": train_dataloader.batch_size,
                "train_dataset_size": len(train_dataloader.dataset),
                "eval_dataset_size": len(eval_dataloader.dataset) if eval_dataloader else 0,
                "device": str(frame.device),
                "module": get_leaf_layers_info(model)
            }
        )
    log_writer = open(frame.log_path, 'w')
    model.to(frame.device)
    load_parameter(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, ck_pth=ck_pth) # 初始化模型

    best_train_accuracy = 0
    best_test_accuracy = 0
    model_save_epoch = 0
    train_loss_note = AverageMeter("Train-Loss", ":.6f")
    train_acc_note = AverageMeter("Acc", ":.4f")
    test_loss_note = AverageMeter("Test-Loss", ":.6f")
    test_acc_note = AverageMeter("Acc", ":.4f")
    progress_writer = ProgressMeter(frame.epochs, len(train_dataloader), 
                                    [train_loss_note, train_acc_note, test_loss_note, test_acc_note],
                                    prefix="Batch")
    DRAW_ORINGINAL_INPUT = True if SWANLAB_AVAILABLE else False
    try:
        for epoch in range(frame.start_epoch+1, frame.epochs+1):
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'\n{formatted_time} Epoch {epoch}:')
            model.train()  # 开启训练模式，自训练没有测试模式，所以这个可以在训练之前设置
            for data, label in tqdm(train_dataloader, total=len(train_dataloader), desc="Training", leave=True):
                batchs = data.size(0)
                data, label = data.to(frame.device), label.to(frame.device)
                optimizer.zero_grad()
                output = model(data)
                loss = frame.loss_func(output, label)
                acc = topk_accuracy(output, label)

                train_loss_note.update(loss.item(), batchs)
                train_acc_note.update(acc[0].item(), batchs)
                loss.backward()
                optimizer.step()

            DRAW_FEATURE_MAPS = True if (epoch % 20 == 0 or epoch == 1) and SWANLAB_AVAILABLE and model.if_draw_feature_map() else False # 每20个epoch绘制一次特征图
            if eval_dataloader is not None:
                DRAW_FEATURE_POSITION = int(frame.feature_map_position * len(eval_dataloader.dataset)) if DRAW_FEATURE_MAPS else -1 # 控制绘制特征图的位置
                REGISTERED_HOOKS = False # 用来控制仅注册一次hook
                all_labels = np.empty((len(eval_dataloader.dataset),), dtype=np.int16)
                all_preds = np.empty((len(eval_dataloader.dataset),), dtype=np.int16)
                idx = 0
                model.eval()
                with torch.no_grad():
                    for data, label in tqdm(eval_dataloader, desc='Testing ', total=len(eval_dataloader), leave=True):
                        batchs = data.size(0)
                        frame.remove_hooks(epoch)  # 移除hook
                        if (idx+batchs) >= DRAW_FEATURE_POSITION: 
                            if DRAW_FEATURE_MAPS and not REGISTERED_HOOKS:
                                frame.register_hooks(model) # 仅注册一次hook
                                REGISTERED_HOOKS = True
                            if DRAW_ORINGINAL_INPUT:
                                max_channel_features = min(data.size(1), frame.feature_map_num)
                                max_batch_features = min(data.size(0), frame.feature_map_num)
                                swanlab.log({f"original_input/single_channel_map": [swanlab.Image(img) for img in data[0, :max_channel_features]]}) # 绘制原始输入图像
                                swanlab.log({f"original_input/batch_rgb": [swanlab.Image(img[:3]) for img in data[:max_batch_features]]}) # 绘制原始输入图像
                                DRAW_ORINGINAL_INPUT = False
                        data, label = data.to(frame.device), label.to(frame.device)
                        output = model(data)
                        _, preds = torch.max(output, 1)
                        all_labels[idx:idx+batchs] = label.cpu().numpy()
                        all_preds[idx:idx+batchs] = preds.cpu().numpy()
                        loss = frame.loss_func(output, label)
                        acc = topk_accuracy(output, label)
                        test_loss_note.update(loss.item(), batchs)
                        test_acc_note.update(acc[0].item(), batchs)
                        idx += batchs

            test_accuracy = test_acc_note.avg
            test_avg_loss = test_loss_note.avg
            train_accuracy = train_acc_note.avg
            train_avg_loss = train_loss_note.avg

            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= frame.min_lr:
                pass
            else:
                if scheduler is not None:
                    scheduler.step()
            result = progress_writer.epoch_summary(epoch, f"Lr: {current_lr:.2e}")
            if SWANLAB_AVAILABLE:
                swanlab.log({"Train/loss":train_avg_loss}, step=epoch)
                swanlab.log({"Train/accuracy":train_accuracy}, step=epoch)
                swanlab.log({"Test/loss":test_avg_loss}, step=epoch)
                swanlab.log({"Test/accuracy":test_accuracy}, step=epoch)
                swanlab.log({"Learning_rate":current_lr}, step=epoch)
            log_writer.write(result + "\n")
            log_writer.flush()
            train_loss_note.reset()
            train_acc_note.reset()
            test_loss_note.reset()
            test_acc_note.reset()
            is_best = False
            if test_accuracy > frame.test_epoch_max_acc: # 使用测试集的预测准确率进行模型保存
                frame.test_epoch_min_loss = test_avg_loss
                frame.test_epoch_max_acc = test_accuracy
                frame.best_all_labels = all_labels
                frame.best_all_preds = all_preds
                best_train_accuracy = train_accuracy
                best_test_accuracy = test_accuracy
                model_save_epoch = epoch
                is_best = True
            elif test_accuracy == frame.test_epoch_max_acc:
                if test_avg_loss < frame.test_epoch_min_loss:
                    frame.test_epoch_min_loss = test_avg_loss
                    frame.test_epoch_max_acc = test_accuracy
                    frame.best_all_labels = all_labels
                    frame.best_all_preds = all_preds
                    best_train_accuracy = train_accuracy
                    best_test_accuracy = test_accuracy
                    model_save_epoch = epoch
                    is_best = True
            save_model(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                       avg_loss=test_avg_loss, avg_acc=test_accuracy, is_best=is_best)
        finish_work()
    except KeyboardInterrupt: # 捕获键盘中断信号
        finish_work()
        print(f"Training interrupted due to: KeyboardInterrupt")
        clean_up(frame=frame)
    except Exception as e: 
        print(f"An error occurred during training: {e}")
        print(traceback.format_exc())  # 打印完整的堆栈跟踪
        clean_up(frame=frame)
    finally:
        sys.exit(0)

def print_report(all_labels, all_preds, log_writer=None):
    accuracy = accuracy_score(all_labels, all_preds)
    clf_report = classification_report(all_labels, all_preds, digits=4)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n{formatted_time} Test Accuracy: {accuracy:.6f}. Cohen's Kappa: {kappa:.4f}")
    print("Classification Report:\n", clf_report)
    print("Confusion Matrix:\n", np.array2string(conf_matrix, separator=', '))

    if log_writer is not None:
        log_writer.write(f"\n\n{formatted_time} Test_acc: {accuracy:.4f}. Cohen's Kappa: {kappa:.4f}\n")
        log_writer.write(f"Classification Report:\n")
        log_writer.write(clf_report + "\n")
        log_writer.write("Confusion Matrix:\n")
        log_writer.write(np.array2string(conf_matrix, separator=', '))
        log_writer.write('\n')
        log_writer.flush()