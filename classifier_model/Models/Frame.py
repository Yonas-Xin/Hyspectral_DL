import os, sys
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import torch
import shutil
from utils import AverageMeter, ProgressMeter, topk_accuracy, copy_config_json_to_output_dir, load_config_json
import logging
from torch.nn import DataParallel # 未来建议升级为 DistributedDataParallel (DDP)
import torch.nn as nn  
import re
from multiprocessing import cpu_count
try: # 使用swanlab进行试验管理
    import swanlab
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

class Classifier_Frame:
    def __init__(self,  model_name, 
                        model, 
                        optimizer, 
                        train_dataloader, 
                        eval_dataloader=None, 
                        test_dataloader=None,
                        scheduler=None, 
                        ck_pth=None, 
                        device=None,
                        augment=None,
                        if_full_cpu=True, 
                        feature_map_layer_n=[], 
                        feature_map_num=12, 
                        feature_map_interval=20,
                        use_data_parallel=False,
                        swanlab_available = False,
                        mode="Classifier",
                        output_dir=None,
                        config_path=None):
        """
        初始化训练框架
        
        Args:
            model_name: 模型名称
            model: 模型
            optimizer: 优化器
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
            test_dataloader: 测试数据加载器
            scheduler: 学习率调度器
            ck_pth: 断点续训路径
            device: 训练设备
            if_full_cpu: 是否全负荷cpu
            feature_map_layer_n: 绘制特征图的层名
            feature_map_num: 绘制特征图的数量
            feature_map_interval: 绘制特征图的间隔
            use_data_parallel: 是否使用DataParallel进行多GPU训练
            mode: 训练模式，默认为 "Classifier"，可选 "Classifier" 或 "Contrastive"
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler
        self.ck_pth = ck_pth
        self.augment = augment
        self.mode = mode
        self.output_dir = output_dir
        self.config_path = config_path
        
        self.loss_func = nn.CrossEntropyLoss()
        self.use_data_parallel = use_data_parallel # 是否使用DataParallel进行多GPU训练
        self.model_name = model_name
        self.logger = None
        self.swanlab_available = swanlab_available and SWANLAB_AVAILABLE   

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else: self.device = device
        self.model.to(self.device)

        # 配置训练信息、用于断点训练
        self.if_full_cpu = if_full_cpu

        self.test_epoch_min_loss = float('inf')
        self.test_epoch_max_acc = -1
        self.start_epoch = 0

        self.best_all_labels = None
        self.best_all_preds = None
        self.best_selection_name = 'Validation' if self.eval_dataloader is not None else 'Training'
        self.final_test_accuracy = None
        self.final_test_loss = None
        self.if_draw_feature_maps = False

        self.BATCH_IMGS = [] # 存储数据，用来绘制特征图
        self.feature_map_layer_n = feature_map_layer_n
        self.feature_map_num = feature_map_num
        self.feature_map_interval = feature_map_interval  # 每隔多少个epoch绘制一次特征图
        if self.augment is not None:
            print(f'INFO: The train dataset will use augmentation.')

    def _resolve_parent_dir(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not self.output_dir:
            return base_dir

        module_dir_name = 'contrastive_learning' if self.mode == 'Contrastive' else 'classifier_model'
        return os.path.join(self.output_dir, module_dir_name)

    def _init_files(self, id=None): # 初始化文件夹和日志文件, id用于swanlab断点续训
        current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
        if id is not None:
            model_save_name = f'{self.model_name}_{current_time}_ID{id}'
        else:
            model_save_name = f'{self.model_name}_{current_time}'
        self.parent_dir = self._resolve_parent_dir()  # 创建一个父目录保存训练结果
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
        copied_config_path = copy_config_json_to_output_dir(self.config_path, self.model_dir)
        if copied_config_path is not None:
            print(f'INFO: Config JSON copied to: {copied_config_path}')
        print(f'INFO: Training outputs will be saved to: {self.model_dir}')
        self._setup_logger()
    
    def _dataparallel(self):
        if self.use_data_parallel and torch.cuda.device_count() > 1:
            print(f"INFO: Use DataParallel, The number of GPUs is: {torch.cuda.device_count()}")
            self.model = DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        else:
            print(f'INFO: Cuda device count: {torch.cuda.device_count()} And the current device:{self.device}') 

    def _setup_logger(self):
        """配置 logging"""
        self.logger = logging.getLogger(self.model_name)
        self.logger.propagate = False # 防止日志重复打印
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = [] # 清除已有的 handler，防止重复打印

        file_handler = logging.FileHandler(self.log_path, mode='w', encoding='utf-8')
        file_formatter = logging.Formatter('%(message)s') # 文件只记录消息内容
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s') # 控制台只显示消息
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _find_id(self, ck_pth): # 从ckpt路径中提取ID
        if ck_pth is None:
            return None
        ck_pth = os.path.basename(ck_pth)
        pattern = r"_ID([^\.]+)\."
        match = re.search(pattern, ck_pth)
        if match:
            return match.group(1)[:21]
        return None

    def _full_cpu(self):
        cpu_num = cpu_count()
        if self.if_full_cpu:
            torch.set_num_threads(cpu_num)
            print('INFO: Using cpu core num: ', cpu_num)

    def _check_input(self, model):
        """检查画特征图的层是否存在于原网络中"""
        if self.feature_map_layer_n is not None: # 检查指定的绘制特征图的层名是否存在于模型中
            for name in self.feature_map_layer_n:
                if name not in dict(model.named_modules()).keys():
                    raise ValueError(f"Layer name '{name}' not found in the model's modules. \
                                        The available layers are:\n {list(model.named_modules().keys())}")
                    
    def _reshape_transform_3d(self, tensor):
        # 输入可能是 [N, C, D, H, W]，对 D 维做平均，返回 [N, C, H, W]
        if tensor.ndim == 5:
            return tensor.mean(dim=2)
        return tensor

    def _init_all(self, experiment_name):
        ID = self._find_id(self.ck_pth) # 从ckpt路径中提取swanlab的ID
        if self.mode == "Contrastive": model = self.model.encoder_q
        else: model = self.model
        if self.swanlab_available:
            self._check_input(model) # 检查模型与参数的兼容性
            config_json = load_config_json(self.config_path)
            swanlab_config = {
                "model_name": type(model).__name__,
                "epochs": self.epochs,
                "batch_size": self.train_dataloader.batch_size,
                "train_dataset_size": len(self.train_dataloader.dataset),
                "eval_dataset_size": len(self.eval_dataloader.dataset) if self.eval_dataloader else 0,
                "test_dataset_size": len(self.test_dataloader.dataset) if self.test_dataloader else 0,
                "device": str(self.device),
                "use_data_parallel": self.use_data_parallel,
                "module": get_leaf_layers_info(model),
                "config_json": config_json,
            }
            run = swanlab.init(
                project=experiment_name,
                experiment_name=f"{type(model).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=swanlab_config,
                resume=True if self.ck_pth is not None else False, # 控制断点续训
                id=ID,
            )
            self._init_files(id=run.id)  # 初始化文件夹和日志文件
        else:
            self._init_files(id=None)
        self._dataparallel() # 初始化dataparallel

    # 这个函数不要动
    def _gradcam_feature_maps(self, model, epoch, if_draw_feature_maps): # 微调模式下不能绘制特征图
        if not ((epoch % self.feature_map_interval == 0 or epoch == 1) and self.swanlab_available \
            and if_draw_feature_maps and self.BATCH_IMGS): return
        if isinstance(model, DataParallel): model = model.module  # 解包DataParallel
        if self.mode == "Contrastive":model = model.encoder_q
        else: # 分类模式下
            if model.lock_grad: # 如果编码器参数被冻结，无法计算GradCAM所需的梯度，因此跳过特征图绘制
                print("INFO: The model is in lock_grad mode, and the feature map visualization will be skipped.")
                return
        train_mode = model.training  # 记录当前模式
        model.eval()  # 切换到评估模式
        data, label = self.BATCH_IMGS[0], self.BATCH_IMGS[1]
        max_batch_features = min(data.size(0), self.feature_map_num)
        data = data[:max_batch_features] # 仅使用前N个图像绘制特征图
        module_dict = dict(model.named_modules())
        if not self.feature_map_layer_n: # 如果目标层为空
            feature_map_layer_n = []
            for name, module in module_dict.items():
                if isinstance(module, (nn.Conv2d, nn.Conv3d)): # 自动寻找卷积层
                    feature_map_layer_n.append(name)
            if len(feature_map_layer_n) > 2: # 如果层数过多，则只取前三层、中间层、最后一层
                self.feature_map_layer_n = [feature_map_layer_n[0], feature_map_layer_n[len(feature_map_layer_n) // 2], feature_map_layer_n[-1]]
            else:
                self.feature_map_layer_n = [] 
        for layer_name in self.feature_map_layer_n: # 对每个指定层绘制GradCAM
            target_layer = [module_dict[layer_name]]
            if self.mode == "Contrastive": target = None
            else: target = [ClassifierOutputTarget(int(label[i])) for i in range(max_batch_features)]
            cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=self._reshape_transform_3d)
            grayscale_cam = cam(input_tensor=data, targets=target)
            # 绘制叠加图
            visualization_list = []
            for i in range(max_batch_features):
                input_img = data[i].cpu().numpy().transpose(1,2,0)  # 转换为 (H,W,C)
                input_img = stretch_img(input_img[:,:,:3])  # 拉伸到0-1范围
                visualization = show_cam_on_image(input_img, grayscale_cam[i], use_rgb=True)
                visualization_list.append(visualization)
            swanlab.log({f"GradCAMgray/{layer_name}": [swanlab.Image((grayscale_cam[i]*255).astype(np.uint8), 
                                                            caption=f"{int(label[i])}") for i in range(max_batch_features)]}, step=epoch)
            swanlab.log({f"GradCAMoverlay/{layer_name}": [swanlab.Image(visualization_list[i], 
                                                            caption=f"{int(label[i])}") for i in range(max_batch_features)]}, step=epoch)
        if train_mode:
            model.train()  # 恢复之前的模式
            
    def _load_parameter(self, model, optimizer, scheduler=None, ck_pth=None): # 加载模型、优化器、调度器
        self._full_cpu() # 打印配置信息
        if ck_pth is not None:
            checkpoint = torch.load(ck_pth, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            self.test_epoch_min_loss = checkpoint.get('best_loss', float('inf'))
            self.test_epoch_max_acc = checkpoint.get('best_acc', -1)
            try:
                optimizer.load_state_dict(checkpoint['optimizer']) # 恢复优化器
                print('INFO: The optimizer state have been loaded!')
                self.start_epoch = checkpoint.get('epoch', -1)
            except(ValueError, RuntimeError):
                print('INFO: The optimizer is incompatible, and the parameters do not match')
            if scheduler and 'scheduler' in checkpoint: # 恢复调度器
                try:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                except (ValueError, RuntimeError):
                    print('INFO: The scheduler is incompatible')
            print(f"INFO: Loaded checkpoint from epoch {self.start_epoch}, current lr {optimizer.param_groups[0]['lr']}")

    def _save_model(self, model, optimizer, scheduler, epoch=None, avg_loss=None, avg_acc=None, is_best=False):
        if isinstance(model, DataParallel):
            model = model.module  # 解包DataParallel
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_loss': avg_loss,
            'best_acc': avg_acc,
            'scheduler': scheduler.state_dict() if scheduler else None,
            'current_lr': optimizer.param_groups[0]['lr']
        }
        if self.mode == "Contrastive":
            state['backbone'] = model.encoder_q.encoder.state_dict()
            
        torch.save(state, self.model_path)
        if is_best:
            shutil.copyfile(self.model_path, self.model_best_path)
            if self.mode == "Contrastive":
                torch.save(model.encoder_q, self.model_best_path_pt)
            else:
                torch.save(model, self.model_best_path_pt)
            print(f"============The best checkpoint saved at epoch {epoch}============")

    def _load_checkpoint_to_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = self.model.module if isinstance(self.model, DataParallel) else self.model
        model.load_state_dict(checkpoint['model'])
        return checkpoint

    def _evaluate_loader_metrics(self, loader, loss_name="Test-Loss"):
        if loader is None:
            return None, None, 0.0, 0.0
        loss_note = AverageMeter(loss_name, ":.6f")
        acc_note = AverageMeter("Acc", ":.4f")
        all_labels, all_preds = self._validate_one_epoch(self.model, loader, loss_note, acc_note, "Testing")
        return all_labels, all_preds, loss_note.avg, acc_note.avg

    def _log_report(self, all_labels, all_preds, dataset_name="Dataset"):
        accuracy = accuracy_score(all_labels, all_preds)
        clf_report = classification_report(all_labels, all_preds, digits=4)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        self.logger.info(f"{dataset_name} Overall Accuracy: {accuracy:.4f}. Cohen's Kappa: {kappa:.4f}.")
        self.logger.info(f"{dataset_name} Classification Report:")
        self.logger.info(clf_report)
        self.logger.info(f"{dataset_name} Confusion Matrix:")
        self.logger.info(np.array2string(conf_matrix, separator=', '))

    def _finish_train(self, epoch):
        # 最终打印一次最佳训练结果，并在swanlab上记录最终结果和特征图
        self.logger.info(f"\n====================Final {self.best_selection_name} Performance====================")
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.best_selection_name == "Training":
            best_result = (
                f"{formatted_time} Model saved at Epoch {self.model_save_epoch}. "
                f"The best training_acc:{self.best_train_accuracy:.4f}%."
            )
        else:
            best_result = (
                f"{formatted_time} Model saved at Epoch {self.model_save_epoch}. "
                f"The best training_acc:{self.best_train_accuracy:.4f}%. "
                f"The best {self.best_selection_name.lower()}_acc:{self.best_test_accuracy:.4f}%."
            )
        if os.path.exists(self.model_best_path):
            self._load_checkpoint_to_model(self.model_best_path)
            self._gradcam_feature_maps(self.model, epoch + 1, self.if_draw_feature_maps)
        self.logger.info(best_result)
        if self.best_all_labels is not None and self.best_all_preds is not None:
            self._log_report(self.best_all_labels, self.best_all_preds, dataset_name=self.best_selection_name)
        
        # 最后在测试集上评估最终模型的性能
        if self.test_dataloader is not None and os.path.exists(self.model_best_path):
            self.logger.info("\n======================Final Test Performance======================")
            test_labels, test_preds, test_loss, test_acc = self._evaluate_loader_metrics(
                self.test_dataloader,
                loss_name="Test-Loss",
            )
            self.final_test_loss = test_loss
            self.final_test_accuracy = test_acc
            self.logger.info(
                f"Final Test Loss: {test_loss:.6f}. Final Test Accuracy: {test_acc:.4f}%."
            )
            if test_labels is not None and test_preds is not None:
                self._log_report(test_labels, test_preds, dataset_name="Test")
            if self.swanlab_available:
                swanlab.log({"Test/final_loss": test_loss}, step=self.model_save_epoch)
                swanlab.log({"Test/final_accuracy": test_acc}, step=self.model_save_epoch)
        print(f"The best pt path: {self.model_best_path_pt}")

    def _train_one_epoch(self, model, optimizer, train_dataloader, train_loss_note, train_acc_note):
        model.train()  # 开启训练模式 
        for data, label in tqdm(train_dataloader, total=len(train_dataloader), desc="Training", leave=True):
            batchs = data.size(0)
            data = data.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            if self.augment is not None:
                data = self.augment(data)
            output = model(data)
            loss = self.loss_func(output, label)
            acc = topk_accuracy(output, label)

            train_loss_note.update(loss.item(), batchs)
            train_acc_note.update(acc[0].item(), batchs)
            loss.backward()
            optimizer.step()
            
    def _validate_one_epoch(self, model, eval_dataloader, test_loss_note, test_acc_note, TQDM_desc="Validate"):
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data, label in tqdm(eval_dataloader, desc=TQDM_desc, total=len(eval_dataloader), leave=True):
                batchs = data.size(0)
                data = data.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                
                output = model(data)
                _, preds = torch.max(output, 1)
                
                all_labels.append(label.cpu())
                all_preds.append(preds.cpu())
                
                loss = self.loss_func(output, label)
                acc = topk_accuracy(output, label)
                test_loss_note.update(loss.item(), batchs)
                test_acc_note.update(acc[0].item(), batchs)
                
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        return all_labels, all_preds

    def train(self, epochs=300, min_lr=1e-7, experiment_name="CNN_Model_Training"):
        self.epochs = epochs
        self.min_lr = min_lr
        IF_DRAW_FEATURE_MAPS = self.model.if_draw_feature_map()
        self.if_draw_feature_maps = IF_DRAW_FEATURE_MAPS
        self.best_selection_name = 'Validation' if self.eval_dataloader is not None else 'Training'
        self._load_parameter(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, ck_pth=self.ck_pth)
        self._init_all(experiment_name)

        self.best_train_accuracy = 0
        self.best_test_accuracy = 0
        self.model_save_epoch = 0
        train_loss_note = AverageMeter("Train-Loss", ":.6f")
        train_acc_note = AverageMeter("Acc", ":.4f")
        selection_loss_name = f"{self.best_selection_name}-Loss"
        selection_loss_note = AverageMeter(selection_loss_name, ":.6f")
        selection_acc_note = AverageMeter("Acc", ":.4f")
        progress_writer = ProgressMeter(
            self.epochs,
            len(self.train_dataloader),
            [train_loss_note, train_acc_note, selection_loss_note, selection_acc_note],
            prefix="Batch"
        )

        self.BATCH_IMGS = []
        if self.test_dataloader is not None and self.swanlab_available and IF_DRAW_FEATURE_MAPS:
            for data, label in self.test_dataloader:
                max_batch_features = min(data.size(0), self.feature_map_num)
                swanlab.log({
                    "original_input": [
                        swanlab.Image(data[i][:3], caption=f"{int(label[i])}")
                        for i in range(max_batch_features)
                    ]
                })
                self.BATCH_IMGS.append(data[:max_batch_features].detach().cpu())
                self.BATCH_IMGS.append(label[:max_batch_features].detach().cpu())
                break

        epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch + 1, self.epochs + 1):
                formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f'\n{formatted_time} Epoch {epoch}:')

                self._train_one_epoch(self.model, self.optimizer, self.train_dataloader, train_loss_note, train_acc_note)
                all_labels, all_preds = None, None
                if self.eval_dataloader is not None:
                    all_labels, all_preds = self._validate_one_epoch(
                        self.model,
                        self.eval_dataloader,
                        selection_loss_note,
                        selection_acc_note,
                    )

                self._gradcam_feature_maps(self.model, epoch, IF_DRAW_FEATURE_MAPS)
                train_accuracy = train_acc_note.avg
                train_avg_loss = train_loss_note.avg
                selection_accuracy = selection_acc_note.avg if self.eval_dataloader is not None else train_accuracy
                selection_avg_loss = selection_loss_note.avg if self.eval_dataloader is not None else train_avg_loss
                if self.eval_dataloader is None:
                    selection_loss_note.update(train_avg_loss, 1)
                    selection_acc_note.update(train_accuracy, 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler is not None:
                    self.scheduler.step()
                result = progress_writer.epoch_summary(epoch, f"Lr: {current_lr:.2e}")
                if self.swanlab_available:
                    swanlab.log({"Train/loss": train_avg_loss}, step=epoch)
                    swanlab.log({"Train/accuracy": train_accuracy}, step=epoch)
                    swanlab.log({f"{self.best_selection_name}/loss": selection_avg_loss}, step=epoch)
                    swanlab.log({f"{self.best_selection_name}/accuracy": selection_accuracy}, step=epoch)
                    swanlab.log({"Learning_rate": current_lr}, step=epoch)
                self.logger.info(result)

                train_loss_note.reset()
                train_acc_note.reset()
                selection_loss_note.reset()
                selection_acc_note.reset()

                is_best = False
                if selection_accuracy > self.test_epoch_max_acc:
                    self.test_epoch_min_loss = selection_avg_loss
                    self.test_epoch_max_acc = selection_accuracy
                    self.best_all_labels = all_labels
                    self.best_all_preds = all_preds
                    self.best_train_accuracy = train_accuracy
                    self.best_test_accuracy = selection_accuracy
                    self.model_save_epoch = epoch
                    is_best = True
                elif selection_accuracy == self.test_epoch_max_acc:
                    if selection_avg_loss < self.test_epoch_min_loss:
                        self.test_epoch_min_loss = selection_avg_loss
                        self.test_epoch_max_acc = selection_accuracy
                        self.best_all_labels = all_labels
                        self.best_all_preds = all_preds
                        self.best_train_accuracy = train_accuracy
                        self.best_test_accuracy = selection_accuracy
                        self.model_save_epoch = epoch
                        is_best = True

                self._save_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    avg_loss=selection_avg_loss,
                    avg_acc=selection_accuracy,
                    is_best=is_best,
                )
            self._finish_train(epoch)
        except KeyboardInterrupt:
            print("\n[Warning] Training interrupted by user. Generating report for the best model so far...")
            self._finish_train(epoch)
    def _train_contrastive_one_epoch(self, model, optimizer, train_dataloader, train_loss_note, top1_acc_note, top5_acc_note, epoch):
        model.train()  # 开启训练模式
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Train Contrastive", leave=True):
            batchs = data.size(0)
            data = data.to(self.device, non_blocking=True)
            with torch.no_grad():
                q = self.augment(data) if self.augment else data
                k = self.augment(data) if self.augment else data
            
            if not self.BATCH_IMGS and self.swanlab_available: # 训练过程中仅运行一次
                max_draw_samples = min(batchs, self.feature_map_num)
                self.BATCH_IMGS.append(data[:max_draw_samples].detach().cpu())
                self.BATCH_IMGS.append(torch.zeros(max_draw_samples).cpu()) # 占位，保持与标签相同的结构
                q_draw = q.detach().cpu()[:max_draw_samples]
                k_draw = k.detach().cpu()[:max_draw_samples]
                swanlab.log(
                {   "samples/original": [swanlab.Image(img[:3]) for img in self.BATCH_IMGS[0][:max_draw_samples]],
                    "samples/augmented_q": [swanlab.Image(img[:3]) for img in q_draw],
                    "samples/augmented_k": [swanlab.Image(img[:3]) for img in k_draw],
                    }, step=epoch)

            optimizer.zero_grad()
            if isinstance(model, DataParallel):
                model.module.update_key_encoder() # 更新key encoder
                q_out, k_out = model(q, k) # 前向计算
                logits, labels = model.module.calculate_logits(q_out, k_out) # 计算logits和labels
            else:
                model.update_key_encoder() # 更新key encoder
                q_out, k_out = model(q, k) # 前向计算
                logits, labels = model.calculate_logits(q_out, k_out) # 计算logits和labels

            loss = self.loss_func(logits, labels)
            acc1, acc5 = topk_accuracy(logits, labels, topk=(1, 5))

            train_loss_note.update(loss.item(), batchs)
            top1_acc_note.update(acc1.item(), batchs)
            top5_acc_note.update(acc5.item(), batchs)
            
            loss.backward()
            optimizer.step()

    def train_contrastive(self, epochs=300, min_lr=1e-7, experiment_name="Contrastive_Model_Training"):
        self.epochs = epochs
        self.min_lr = min_lr
        IF_DRAW_FEATURE_MAPS = self.model.encoder_q.if_draw_feature_maps

        self._load_parameter(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, ck_pth=self.ck_pth) # 初始化模型
        self._init_all(experiment_name)

        self.best_train_accuracy = 0
        self.model_save_epoch = 0
        self.test_epoch_min_loss = float('inf')
        self.test_epoch_max_acc = -1

        train_loss_note = AverageMeter("Train-Loss", ":.6f")
        top1_acc_note = AverageMeter("Top1-Acc", ":.4f")
        top5_acc_note = AverageMeter("Top5-Acc", ":.4f")
        
        progress_writer = ProgressMeter(self.epochs, len(self.train_dataloader), 
                                        [train_loss_note, top1_acc_note, top5_acc_note],
                                        prefix="Batch")
        self.BATCH_IMGS = []
        try:
            for epoch in range(self.start_epoch + 1, self.epochs + 1):
                formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f'\n{formatted_time} Epoch {epoch}:')
                
                self._train_contrastive_one_epoch(self.model, self.optimizer, self.train_dataloader, train_loss_note, top1_acc_note, top5_acc_note, epoch)
                self._gradcam_feature_maps(self.model, epoch, IF_DRAW_FEATURE_MAPS)
                train_accuracy = top1_acc_note.avg
                train_avg_loss = train_loss_note.avg
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler is not None:
                    self.scheduler.step()
                result = progress_writer.epoch_summary(epoch, f"Lr: {current_lr:.2e}")
                if self.swanlab_available:
                    swanlab.log({"Train/loss":train_avg_loss}, step=epoch)
                    swanlab.log({"Train/Top1_accuracy":train_accuracy}, step=epoch)
                    swanlab.log({"Train/Top5_accuracy":top5_acc_note.avg}, step=epoch)
                    swanlab.log({"Learning_rate":current_lr}, step=epoch)
                self.logger.info(result)

                is_best = False
                if train_accuracy > self.test_epoch_max_acc: # 使用训练集的top1准确率进行模型保存
                    self.test_epoch_min_loss = train_avg_loss
                    self.test_epoch_max_acc = train_accuracy
                    self.best_train_accuracy = train_accuracy
                    self.model_save_epoch = epoch
                    is_best = True
                elif train_accuracy == self.test_epoch_max_acc:
                    if train_avg_loss < self.test_epoch_min_loss:
                        self.test_epoch_min_loss = train_avg_loss
                        self.test_epoch_max_acc = train_accuracy
                        self.best_train_accuracy = train_accuracy
                        self.model_save_epoch = epoch
                        is_best = True
                        
                self._save_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, epoch=epoch, 
                            avg_loss=train_avg_loss, avg_acc=train_accuracy, is_best=is_best)
                            
                train_loss_note.reset()
                top1_acc_note.reset()
                top5_acc_note.reset()
                self.train_dataloader.dataset.reset() if hasattr(self.train_dataloader.dataset, 'reset') else None
                
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            best_result = f'{formatted_time} Model saved at Epoch {self.model_save_epoch}. \
The best top1-acc:{self.best_train_accuracy:.4f}. The best training_loss:{self.test_epoch_min_loss:.6f}.'
            self.logger.info(best_result)

            
        except KeyboardInterrupt:
            print("\n[Warning] Training interrupted by user. Generating report for the best model so far...")
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            best_result = f'{formatted_time} Model saved at Epoch {self.model_save_epoch}. \
The best top1-acc:{self.best_train_accuracy:.4f}. The best training_loss:{self.test_epoch_min_loss:.6f}.'
            self.logger.info(best_result)

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
