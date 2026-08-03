import argparse, json, os, sys, re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from datetime import datetime
from multiprocessing import cpu_count
try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    gdal = None
cpu_num = cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import utils
from core import Hyperspectral_Image
from gdal_utils import Mianvector2mask
from classifier_model.Models.Data import Predict_Shared_Dataset
from classifier_model.Models.Transform import SpectralL2NormPerPixel
from utils import load_config_json
from gdal_utils import NP2GDAL_TYPE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'assets', 'configs', 'Classifier_Predict.json')
CLASSIFIER_PREDICT_CONFIG_HELP = """
[Classifier_Predict.json 参数说明]
- input_data: 待预测高光谱影像路径
- model_pt: 训练好的模型路径（*.pt）
- patch_size: patch 尺寸；null 时从模型名 PatchXX 自动解析
- output_path: 输出 tif 路径；null 时按输入目录自动生成
- output_probability: 是否额外输出类别后验概率栅格（多波段 float32，波段数=类别数 C，softmax；NoData=-1）
- image_block_size: 分块预测块大小
- batch_size: 预测 batch 大小
- device: 设备（auto/cuda/cpu）
- background_value: 背景/无效值
- append_timestamp: 输出名是否追加时间戳
- transform.name: 预测输入变换（如 SpectralL2NormPerPixel；null 关闭）
- dataloader.num_workers: DataLoader worker 数；null 自动计算
- dataloader.num_workers_divisor: 自动 worker 计算分母
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classifier prediction entrypoint.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=CLASSIFIER_PREDICT_CONFIG_HELP,
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
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def build_transform(transform_cfg):
    name = (transform_cfg or {}).get('name')
    if name in (None, '', 'None'):
        return None
    if name == 'SpectralL2NormPerPixel':
        return SpectralL2NormPerPixel()
    raise ValueError(f'Unsupported transform name: {name}')


def build_dataloader(dataset, batch_size, num_workers):
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': bool(num_workers > 0),
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return DataLoader(**kwargs)


def load_ignore_mask(mask_path, raster_ref, rows, cols):
    if mask_path in (None, '', 'None'):
        return None
    mask_array, _, _ = Mianvector2mask(mask_path, tif_path=raster_ref, fill_value=1)
    if mask_array.shape != (rows, cols):
        raise ValueError(
            f'Mask size mismatch: mask={mask_array.shape}, '
            f'image={(rows, cols)}'
        )
    ignore_mask = mask_array != 0
    print(f'[INFO] Rasterized vector mask: {mask_path} | masked pixels={int(np.sum(ignore_mask))}')
    return ignore_mask


def shutdown_loader(loader):
    if loader is None:
        return
    try:
        it = getattr(loader, '_iterator', None)
        if it is not None and hasattr(it, '_shutdown_workers'):
            it._shutdown_workers()
    except Exception:
        pass


def create_incremental_tif(output_path, rows, cols, geotransform, projection, background_value, dtype=np.uint8, bands=1):
    """Create a GeoTIFF initialized with background_value; return (dataset, band1).

    bands=1 时写单波段（分类图）；bands>1 时写多波段（如概率栅格）。
    """
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(output_path, cols, rows, bands, NP2GDAL_TYPE[np.dtype(dtype)],
                       options=['INTERLEAVE=PIXEL'])
    if ds is None:
        raise IOError(f'Cannot create output file: {output_path}')
    if geotransform is not None:
        ds.SetGeoTransform(geotransform)
    if projection is not None:
        ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    for b in range(1, bands + 1):
        ds.GetRasterBand(b).SetNoDataValue(background_value)

    # 建立色表
    if dtype == np.uint8 and bands == 1:
        try:
            hex_colors = utils.get_full_academic_color_256()
            rgb_colors = [tuple(utils.hex_to_rgb(color)) for color in hex_colors]
            max_val = min(len(rgb_colors), 255)
            color_table = gdal.ColorTable()
            
            # 默认全部初始化为透明白色 (255, 255, 255, 0)
            for i in range(256):
                color_table.SetColorEntry(i, (255, 255, 255, 0))
                
            # 设置分类颜色 (RGBA)
            for val in range(max_val):
                if val < 254:  # 保留 254 和 255 作为特殊掩膜/背景值
                    r, g, b = rgb_colors[val]
                    color_table.SetColorEntry(int(val), (r, g, b, 255))
            
            # 特殊值设定
            # 254: 外部掩膜，设为灰色
            color_table.SetColorEntry(254, (128, 128, 128, 255))
            # 255: 无效背景值，设为白色透明
            color_table.SetColorEntry(255, (255, 255, 255, 0))
            
            band.SetRasterColorTable(color_table)
            band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        except Exception as e:
            print(f"[WARN] Failed to create color table for TIFF: {e}")
    if bands == 1:
        init_data = np.full((rows, cols), background_value, dtype=dtype)
        band.WriteArray(init_data)
    else:
        init_data = np.full((bands, rows, cols), background_value, dtype=dtype)
        ds.WriteArray(init_data)
    ds.FlushCache()
    return ds, band


def infer_patch_size(model_path, patch_size_override):
    if patch_size_override is not None:
        return int(patch_size_override)

    patch_size_match = re.search(r'Patch(\d+)', os.path.basename(model_path), re.I)
    if patch_size_match is None:
        raise ValueError("Patch size not found in model path. Set patch_size in config.")
    return int(patch_size_match.group(1))


def resolve_output_path(config, model_path):
    output_path = config.get('output_path')
    if output_path:
        base_output = output_path
    else:
        model_name = os.path.basename(model_path).split('.pt')[0]
        model_name = os.path.basename(model_name).split('_ID')[0]
        base_output = os.path.join(os.path.dirname(model_path), f'{model_name}_PRE.tif')

    if config.get('append_timestamp', True):
        current_time = datetime.now().strftime('%Y%m%d%H%M')
        return f"{base_output[:-4]}_{current_time}.tif"
    return base_output


def main():
    args = parse_args()
    config = load_config(args.config)

    cpu_num = cpu_count()

    input_data = config['input_data']
    model_pt = config['model_pt']
    image_block_size = int(config['image_block_size'])
    batch_size = int(config['batch_size'])

    patch_size = infer_patch_size(model_pt, config.get('patch_size'))
    print(f'The Patch Size: {patch_size}')

    output_path = resolve_output_path(config, model_pt)

    device = resolve_device(config.get('device', 'auto'))

    img = Hyperspectral_Image(input_data, init_fig=True)
    external_mask_value = 254
    ignore_mask = load_ignore_mask(config.get('mask_path'), img.dataset, img.rows, img.cols)

    background_value = int(config.get('background_value', 255))
    output_probability = bool(config.get('output_probability', False))
    PROB_NODATA = -1.0

    model = torch.load(model_pt, weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    num_classes = None
    if output_probability:
        with torch.no_grad():
            dummy = torch.zeros((1, img.bands, patch_size, patch_size), dtype=torch.float32, device=device)
            num_classes = int(model(dummy).shape[1])
        print(f'[INFO] Probability raster enabled | num_classes={num_classes}')

    transform = build_transform(config.get('transform'))

    dataloader_cfg = config.get('dataloader', {})
    base_workers = dataloader_cfg.get('num_workers')
    if base_workers is None:
        divisor = max(1, int(dataloader_cfg.get('num_workers_divisor', 4)))
        base_workers = cpu_num // divisor

    dataset = Predict_Shared_Dataset(
        patch_size=patch_size,
        bands=img.bands,
        max_block_size=image_block_size,
        transform=transform,
    )

    out_ds, out_band = create_incremental_tif(
        output_path, img.rows, img.cols, img.geotransform, img.projection, background_value
    )
    out_prob_ds = None
    if output_probability:
        prob_path = output_path.replace('.tif', '_PROB.tif')
        out_prob_ds, _ = create_incremental_tif(
            prob_path, img.rows, img.cols, img.geotransform, img.projection,
            PROB_NODATA, dtype=np.float32, bands=num_classes,
        )
        print(f'[INFO] The probability output: {prob_path}')

    dataloader = None
    current_workers = None
    print(f'[INFO] The output: {output_path}')
    with torch.no_grad():
        for image_block, background_mask, i, j in img.image_block_iter(
            block_size=image_block_size,
            patch_size=patch_size,
        ):
            rows, cols = background_mask.shape
            predict_map = np.zeros((rows, cols), dtype=np.uint8) + background_value
            if ignore_mask is not None:
                block_ignore_mask = ignore_mask[i : i + rows, j : j + cols]
                valid_external_mask = background_mask & block_ignore_mask
                predict_map[valid_external_mask] = external_mask_value
                background_mask = background_mask & ~block_ignore_mask
            if not np.any(background_mask):
                out_band.WriteArray(predict_map, xoff=j, yoff=i)
                out_ds.FlushCache()
                
                continue

            dataset.update_data(image_block, background_mask)
            target_workers = base_workers

            if dataloader is None or current_workers != target_workers:
                shutdown_loader(dataloader)
                dataloader = build_dataloader(dataset, batch_size=batch_size, num_workers=target_workers)
                current_workers = target_workers
                print(f'[INFO] Switch num_workers -> {current_workers}')

            idx = 0
            predict_data = torch.empty(len(dataset), dtype=torch.uint8, device=device)
            prob_data = torch.empty((len(dataset), num_classes), dtype=torch.float32, device=device) if output_probability else None
            for data in tqdm(dataloader, total=len(dataloader), desc=f'Block Predicting(nw={current_workers})'):
                bsz = data.shape[0]
                data = data.to(device, non_blocking=(current_workers > 0))
                outputs = model(data)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                _, predicted = torch.max(outputs, 1)
                predict_data[idx : idx + bsz] = predicted
                if output_probability:
                    prob_data[idx : idx + bsz] = torch.softmax(outputs, dim=1)
                idx += bsz

            pred_np = predict_data.cpu().numpy() if predict_data.device.type == 'cuda' else predict_data.numpy()
            predict_map[background_mask] = pred_np
            out_band.WriteArray(predict_map, xoff=j, yoff=i)
            out_ds.FlushCache()

            if output_probability:
                prob_np = prob_data.cpu().numpy()  # [N, C]
                prob_map = np.full((num_classes, rows, cols), PROB_NODATA, dtype=np.float32)
                prob_map[:, background_mask] = prob_np.T  # [C, N] -> 填充有效像素
                out_prob_ds.WriteArray(prob_map, xoff=j, yoff=i)
                out_prob_ds.FlushCache()

    shutdown_loader(dataloader)
    out_band = None
    out_ds = None
    if out_prob_ds is not None:
        out_prob_ds = None

    print(f'Prediction completed. Output saved to: {output_path}')
    if output_probability:
        print(f'Probability raster saved to: {prob_path}')


if __name__ == '__main__':
    main()
