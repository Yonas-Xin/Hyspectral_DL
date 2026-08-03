import argparse, json, os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from multiprocessing import cpu_count
cpu_num = cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from classifier_model.Models.Augment import HighDimBatchAugment
from classifier_model.Models.Data import Contrastive_Dataset, MIRBS_Dataset
from classifier_model.Models.Frame import Classifier_Frame
from classifier_model.Models.Models import Ete_Model, Moco_Model
from classifier_model.Models.Scheduler import WarmupLinearSchedule
from utils import load_config_json, search_files_in_directory, resolve_optional_path

TRAIN_DICT = {
    'ETE': Ete_Model,
    'MOCO': Moco_Model,
}
DATASET_DICT = {
    1: Contrastive_Dataset,
    2: MIRBS_Dataset,
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'assets', 'configs', 'Contrastive_Train.json')
CONTRASTIVE_TRAIN_CONFIG_HELP = """
[Contrastive_Train.json 参数说明]
- encoder_model_name: 编码器模型名称（从classifier_model\\Models\\Encoder.py注册器中查看）
- train_mode: 对比学习模式（ETE 或 MOCO）
- data_manage_mode: 数据管理模式（1=样本列表，2=多图动态采样，即原始影像列表）
- images_dir: 训练影像目录
- patch_size: patch 尺寸，当 data_manage_mode=2 时必需
- ck_pth: 断点续训 checkpoint 路径
- epochs: 主训练轮数
- warm_up_epochs: 预热轮数（总训练轮数 = epochs + warm_up_epochs）
- batch_size: batch 大小
- init_lr: 初始学习率
- min_lr: 最低学习率
- output_dir: 训练结果根目录；可填绝对路径，或相对项目根目录的路径
- experiment_name: 实验名（swanlab 项目名）
- model_name: 模型保存名称；null 则自动生成
- swanlab_available: 是否启用 swanlab
- if_full_cpu: 是否启用 full CPU 线程
- use_data_parallel: 是否启用 DataParallel（<2 卡会自动关闭）
- multithreading_mode: 是否启用多线程数据加载
- device: 设备（auto/cuda:0/cpu）
- image_extensions: 搜索数据文件后缀列表
- K/m/T: 对比学习超参数（队列长度/动量/温度）
- display_nums: 预留显示参数
- optimizer.name: 优化器名称（Adam/AdamW）
- optimizer.weight_decay: 权重衰减
- dataloader.num_workers: DataLoader worker 数；null 自动计算
- dataloader.num_workers_divisor: 自动 worker 计算分母
- dataloader.pin_memory: 是否 pin_memory
- dataloader.prefetch_factor: 预取因子（num_workers>0 生效）
- dataloader.persistent_workers: 是否持久 worker（num_workers>0 生效）
- dataloader.drop_last: 是否丢弃最后不完整 batch
- augment.enabled: 是否启用批量增强
- augment.*: 数据增强参数（光谱掩膜、band dropout、噪声、擦除等）
- feature_map.layer_names: GradCAM 指定层名列表；空列表自动选择，null 不画
- feature_map.num: 每次可视化样本数
- feature_map.interval: 可视化间隔 epoch
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Contrastive training entrypoint.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=CONTRASTIVE_TRAIN_CONFIG_HELP,
    )
    parser.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to json config. Default: {DEFAULT_CONFIG_PATH}',
    )
    return parser.parse_args()


def load_config(config_path):
    return load_config_json(config_path)


def resolve_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)

def build_optimizer(model, optimizer_cfg, init_lr):
    name = optimizer_cfg.get('name', 'Adam')
    weight_decay = optimizer_cfg.get('weight_decay', 0.0)
    if name == 'Adam':
        return optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    if name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer name: {name}')


def build_augment(augment_cfg):
    if not augment_cfg.get('enabled', True):
        return None

    return HighDimBatchAugment(
        spectral_mask_prob=augment_cfg.get('spectral_mask_prob', 0.5),
        spectral_mask_p=augment_cfg.get('spectral_mask_p', 0.25),
        band_dropout_prob=augment_cfg.get('band_dropout_prob', 0.5),
        bands_dropout_p=augment_cfg.get('bands_dropout_p', 0.25),
        flip_prob=augment_cfg.get('flip_prob', 0.0),
        rotate_prob=augment_cfg.get('rotate_prob', 0.0),
        add_gaussian_prob=augment_cfg.get('add_gaussian_prob', 0.0),
        erase_prob=augment_cfg.get('erase_prob', 0.0),
        rotate_degrees=augment_cfg.get('rotate_degrees', 90.0),
        noise_std=augment_cfg.get('noise_std', 0.01),
        mean=augment_cfg.get('mean', 0.0),
        erase_scale=tuple(augment_cfg.get('erase_scale', [0.01, 0.3])),
        erase_ratio=tuple(augment_cfg.get('erase_ratio', [0.4, 2.5])),
    )


def resolve_dataloader_workers(cpu_num, dataloader_cfg, multithreading_mode):
    if not multithreading_mode:
        return 0

    workers = dataloader_cfg.get('num_workers')
    if workers is None:
        divisor = max(1, int(dataloader_cfg.get('num_workers_divisor', 4)))
        workers = cpu_num // divisor
    return workers


def main():
    args = parse_args()
    config_path = resolve_optional_path(args.config, PROJECT_ROOT)
    config = load_config(config_path)
    output_dir = resolve_optional_path(config.get('output_dir'), PROJECT_ROOT)

    cpu_num = cpu_count()

    device = resolve_device(config.get('device', 'auto'))
    gpu_count = torch.cuda.device_count()

    use_data_parallel = bool(config.get('use_data_parallel', True))
    if gpu_count < 2:
        use_data_parallel = False

    train_mode = config['train_mode']
    if train_mode not in TRAIN_DICT:
        raise ValueError(f'Unsupported train_mode: {train_mode}')

    data_manage_mode = int(config['data_manage_mode'])
    if data_manage_mode not in DATASET_DICT:
        raise ValueError(f'Unsupported data_manage_mode: {data_manage_mode}')

    multithreading_mode = bool(config.get('multithreading_mode', True))
    patch_size = int(config['patch_size'])
    image_extensions = tuple(config.get('image_extensions', ['tif', 'dat', 'bin']))
    image_lists = search_files_in_directory(config['images_dir'], image_extensions)

    dataset = DATASET_DICT[data_manage_mode](
        image_lists,
        patch_size=patch_size,
        multith_mode=multithreading_mode,
    )

    model = TRAIN_DICT[train_mode](
        encoder_model_name=config['encoder_model_name'],
        in_shape=dataset.data_shape,
        K=int(config.get('K', 65536)),
        m=float(config.get('m', 0.999)),
        T=float(config.get('T', 0.07)),
    )

    epochs = int(config['epochs'])
    warm_up_epochs = int(config.get('warm_up_epochs', 0))
    init_lr = float(config['init_lr'])
    min_lr = float(config['min_lr'])

    optimizer = build_optimizer(model, config.get('optimizer', {}), init_lr)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=warm_up_epochs,
        t_total=epochs + warm_up_epochs,
        min_lr=min_lr,
    )

    augment = build_augment(config.get('augment', {}))

    dataloader_cfg = config.get('dataloader', {})
    dataloader_num_workers = resolve_dataloader_workers(cpu_num, dataloader_cfg, multithreading_mode)

    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': int(config['batch_size']),
        'shuffle': True,
        'pin_memory': dataloader_cfg.get('pin_memory', True),
        'num_workers': dataloader_num_workers,
        'drop_last': bool(dataloader_cfg.get('drop_last', True)),
    }

    if dataloader_num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = dataloader_cfg.get('prefetch_factor', 2)
        dataloader_kwargs['persistent_workers'] = dataloader_cfg.get('persistent_workers', True)

    dataloader = DataLoader(**dataloader_kwargs)

    feature_map_cfg = config.get('feature_map', {})
    model_name = config.get('model_name')
    if not model_name:
        model_name = f"{train_mode}_{config['encoder_model_name']}"

    print(f'🔍 PyTorch Version: {torch.__version__}')
    print(f'📊 Using num_workers: {dataloader_num_workers}')
    print(f'🎯 Image shape: {dataset.data_shape}')

    frame = Classifier_Frame(
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        scheduler=scheduler,
        ck_pth=config.get('ck_pth'),
        device=device,
        augment=augment,
        if_full_cpu=bool(config.get('if_full_cpu', True)),
        feature_map_layer_n=feature_map_cfg.get('layer_names', []),
        feature_map_num=int(feature_map_cfg.get('num', 36)),
        feature_map_interval=int(feature_map_cfg.get('interval', 10)),
        use_data_parallel=use_data_parallel,
        swanlab_available=bool(config.get('swanlab_available', False)),
        mode='Contrastive',
        output_dir=output_dir,
        config_path=config_path,
    )

    frame.train_contrastive(
        epochs=epochs + warm_up_epochs,
        min_lr=min_lr,
        experiment_name=config.get('experiment_name', 'Contrastive_Learning_Training'),
    )


if __name__ == '__main__':
    main()
