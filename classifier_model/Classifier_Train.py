import argparse, random, json, os, sys
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
from classifier_model.Models.Data import CNN_Dataset
from classifier_model.Models.Frame import Classifier_Frame
from classifier_model.Models.Models import MODEL_REGISTRY
from classifier_model.Models.Scheduler import WarmupLinearSchedule
from classifier_model.Models.Transform import SpectralL2NormPerPixel
from utils import load_config_json, read_dataset_from_txt, resolve_optional_path

"""
必须提供训练集，验证集与测试集可选；验证集用于训练过程中模型选优，测试集用于训练结束后最终评估
如果没有验证集，训练过程中使用训练集评估并选优；如果没有测试集，训练结束后不做最终评估
"""

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'assets', 'configs', 'Classifier_Train.json')
CLASSIFIER_TRAIN_CONFIG_HELP = """
[Classifier_Train.json 参数说明]
- model_selected: 选择分类模型名称（可选模型从 classifier_model\\Models\\Models.py 注册表查看）
- meta: 裁剪生成的 .meta.json 路径；填了则以 meta 为准（数据集与 out_classes 取自 meta，忽略下方 *_images_txt），null 则按下方 txt 方式配置
- train_images_txt: 训练集列表 txt 路径（meta 为 null 时使用）
- val_images_txt: 验证集列表 txt 路径；训练过程中用于模型选优（meta 为 null 时使用）
- test_images_txt: 测试集列表 txt 路径；训练结束后使用最佳模型做最终评估（meta 为 null 时使用）
- out_classes: 类别数；null 表示自动从数据读取（meta 提供 out_classes 时以 meta 为准）
- pretrain_pth: 预训练权重路径（用于加载 backbone）
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
- device: 设备（auto/cuda:0/cpu）
- shuffle_seed: 验证/测试集打乱随机种子
- transform.name: 输入变换（如 SpectralL2NormPerPixel；null 关闭）
- optimizer.name: 优化器名称（AdamW/Adam）
- optimizer.weight_decay: 权重衰减
- dataloader.num_workers: DataLoader worker 数；null 自动计算
- dataloader.num_workers_divisor: 自动 worker 计算分母
- dataloader.pin_memory: 是否 pin_memory
- dataloader.prefetch_factor: 预取因子（num_workers>0 生效）
- dataloader.persistent_workers: 是否持久 worker（num_workers>0 生效）
- augment.enabled: 是否启用批量增强
- augment.*: 数据增强参数（翻转、旋转、噪声、擦除、光谱掩膜等）
- feature_map.layer_names: GradCAM 指定层名列表；空列表自动选择；null 不画
- feature_map.num: 每次可视化样本数
- feature_map.interval: 可视化间隔 epoch
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classifier training entrypoint.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=CLASSIFIER_TRAIN_CONFIG_HELP,
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


def build_transform(transform_cfg):
    name = (transform_cfg or {}).get('name')
    if name in (None, '', 'None'):
        return None
    if name == 'SpectralL2NormPerPixel':
        return SpectralL2NormPerPixel()
    raise ValueError(f'Unsupported transform name: {name}')


def build_optimizer(model, optimizer_cfg, init_lr):
    name = optimizer_cfg.get('name', 'AdamW')
    weight_decay = optimizer_cfg.get('weight_decay', 0.0)
    if name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    if name == 'Adam':
        return optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer name: {name}')


def build_augment(augment_cfg):
    if not augment_cfg.get('enabled', True):
        return None

    return HighDimBatchAugment(
        spectral_mask_prob=augment_cfg.get('spectral_mask_prob', 0.0),
        band_dropout_prob=augment_cfg.get('band_dropout_prob', 0.0),
        erase_prob=augment_cfg.get('erase_prob', 0.0),
        noise_std=augment_cfg.get('noise_std', 0.0),
        mean=augment_cfg.get('mean', 0.0),
        spectral_mask_p=augment_cfg.get('spectral_mask_p', 0.25),
        bands_dropout_p=augment_cfg.get('bands_dropout_p', 0.25),
        flip_prob=augment_cfg.get('flip_prob', 0.0),
        rotate_prob=augment_cfg.get('rotate_prob', 0.0),
        add_gaussian_prob=augment_cfg.get('add_gaussian_prob', 0.0),
        rotate_degrees=augment_cfg.get('rotate_degrees', 90.0),
        erase_scale=tuple(augment_cfg.get('erase_scale', [0.01, 0.3])),
        erase_ratio=tuple(augment_cfg.get('erase_ratio', [0.4, 2.5])),
    )


def build_dataloader(dataset, batch_size, shuffle, dataloader_cfg, fallback_workers):
    num_workers = dataloader_cfg.get('num_workers')
    if num_workers is None:
        divisor = max(1, int(dataloader_cfg.get('num_workers_divisor', 4)))
        num_workers = fallback_workers // divisor

    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'pin_memory': dataloader_cfg.get('pin_memory', True),
        'num_workers': num_workers,
    }

    if num_workers > 0:
        kwargs['prefetch_factor'] = dataloader_cfg.get('prefetch_factor', 2)
        kwargs['persistent_workers'] = dataloader_cfg.get('persistent_workers', True)

    return DataLoader(**kwargs)


def load_dataset_list(txt_path, base_dir):
    resolved_path = resolve_optional_path(txt_path, base_dir)
    if resolved_path is None:
        return []
    return read_dataset_from_txt(resolved_path)


def load_meta_sources(meta_path, base_dir):
    """从裁剪 meta(.json) 解析出 train/val/test 列表路径与 out_classes。

    meta 内的 datasets 路径相对 meta 文件所在目录解析，整个数据目录迁移后仍可用。
    返回 (sources, out_classes)，sources 形如 {'train': 绝对路径, 'val': ..., 'test': ...}。
    """
    resolved_meta = resolve_optional_path(meta_path, base_dir)
    if resolved_meta is None or not os.path.isfile(resolved_meta):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    meta = load_config_json(resolved_meta)
    meta_dir = os.path.dirname(resolved_meta)
    datasets = meta.get('datasets', {}) or {}
    sources = {key: resolve_optional_path(value, meta_dir) for key, value in datasets.items()}
    if 'out_dir' in meta: # meta 中的 out_dir 也相对解析
        sources['out_dir'] = resolve_optional_path(meta.get('out_dir'), meta_dir)
    return sources, meta.get('out_classes')


def infer_num_classes(data_list):
    labels = []
    for line in data_list:
        parts = line.rsplit(maxsplit=1)
        if len(parts) < 2:
            raise ValueError("Dataset txt must include labels in each line: 'path label'.")
        labels.append(int(parts[-1]))
    # 标签必须是从0开始连续的整数
    unique_labels = sorted(set(labels))
    if unique_labels and unique_labels[0] != 0:
        raise ValueError(f"Labels must start from 0, got min label {unique_labels[0]}.")
    expected_labels = list(range(len(unique_labels)))
    if unique_labels != expected_labels:
        raise ValueError(f"Labels must be continuous without gaps: expected {expected_labels}, got {unique_labels}.")
    return len(unique_labels) # 类别数为标签的唯一值数量


def main():
    args = parse_args()
    config_path = resolve_optional_path(args.config, PROJECT_ROOT)
    config = load_config(config_path)

    cpu_num = cpu_count()

    device = resolve_device(config.get('device', 'auto'))
    gpu_count = torch.cuda.device_count()

    use_data_parallel = bool(config.get('use_data_parallel', False))
    if gpu_count < 2:
        use_data_parallel = False

    transform = build_transform(config.get('transform'))

    # 给了 meta 则以 meta 为准；否则按 *_images_txt 配置（原有方式）
    meta_path = config.get('meta')
    meta_out_classes = None
    if meta_path:
        sources, meta_out_classes = load_meta_sources(meta_path, PROJECT_ROOT)
        train_src = sources.get('train')
        val_src = sources.get('val')
        test_src = sources.get('test')
        output_dir = resolve_optional_path(sources.get('out_dir'), PROJECT_ROOT)
    else:
        train_src = config.get('train_images_txt')
        val_src = config.get('val_images_txt')
        test_src = config.get('test_images_txt')
        output_dir = resolve_optional_path(config.get('out_dir'), PROJECT_ROOT)


    train_image_lists = load_dataset_list(train_src, PROJECT_ROOT)
    val_image_lists = load_dataset_list(val_src, PROJECT_ROOT)
    test_image_lists = load_dataset_list(test_src, PROJECT_ROOT)

    if not train_image_lists:
        raise ValueError('Train dataset is empty or not set (check meta or train_images_txt).')

    shuffle_seed = int(config.get('shuffle_seed', 42))
    if val_image_lists:
        random.Random(shuffle_seed).shuffle(val_image_lists)
    if test_image_lists:
        random.Random(shuffle_seed + 1).shuffle(test_image_lists)

    train_dataset = CNN_Dataset(train_image_lists, transform=transform)
    val_dataset = CNN_Dataset(val_image_lists, transform=transform) if val_image_lists else None
    test_dataset = CNN_Dataset(test_image_lists, transform=transform) if test_image_lists else None

    out_classes = config.get('out_classes')
    if meta_path and meta_out_classes is not None:
        out_classes = meta_out_classes
    if out_classes is None:
        out_classes = infer_num_classes(train_image_lists)

    model = MODEL_REGISTRY[config['model_selected']](
        out_classes=out_classes,
        in_shape=train_dataset.data_shape,
    )

    pretrain_pth = config.get('pretrain_pth')
    if pretrain_pth:
        state_dict = torch.load(pretrain_pth, map_location=device)['backbone']
        model.load_encoder_params(state_dict)
        model.freeze_encoder()

    init_lr = float(config['init_lr'])
    min_lr = float(config['min_lr'])
    epochs = int(config['epochs'])
    warm_up_epochs = int(config.get('warm_up_epochs', 0))

    optimizer = build_optimizer(model, config.get('optimizer', {}), init_lr)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=warm_up_epochs,
        t_total=epochs + warm_up_epochs,
        min_lr=min_lr,
    )

    augment = build_augment(config.get('augment', {}))

    dataloader_cfg = config.get('dataloader', {})
    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        dataloader_cfg=dataloader_cfg,
        fallback_workers=cpu_num,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=int(config['batch_size']),
            shuffle=False,
            dataloader_cfg=dataloader_cfg,
            fallback_workers=cpu_num,
        )
    test_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size=int(config['batch_size']), shuffle=False, num_workers=0)

    feature_map_cfg = config.get('feature_map', {})
    model_name = config.get('model_name')
    if not model_name:
        model_name = f"{config['model_selected']}_Patch{train_dataset.data_shape[-1]}"

    print(f'🔍 PyTorch Version: {torch.__version__}')
    print(f'📊 Using num_workers: {train_dataloader.num_workers}')
    print(f'🎯 Image shape: {train_dataset.data_shape}')
    print(f'🔢 Number of classes: {out_classes}')
    print(f'🧪 Dataset sizes - train: {len(train_dataset)}, val: {len(val_dataset) if val_dataset is not None else 0}, test: {len(test_dataset) if test_dataset is not None else 0}')

    frame = Classifier_Frame(
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
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
        mode='Classifier',
        output_dir=output_dir,
        config_path=config_path,
    )

    frame.train(
        epochs=epochs + warm_up_epochs,
        min_lr=min_lr,
        experiment_name=config.get('experiment_name', 'Classifier_Training'),
    )


if __name__ == '__main__':
    main()
