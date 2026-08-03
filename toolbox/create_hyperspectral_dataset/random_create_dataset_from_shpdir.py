import os
import sys
import json
from pathlib import Path
import numpy as np
from osgeo import gdal, ogr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdal_utils import (
    Mianvector2mask, _write_feature_subset_to_shp, batch_random_split_shp,
    clip_by_multishp, build_label_map_from_shps, resolve_source_name,
)
from utils import search_files_in_directory

"""
Split a directory of per-class shapefiles into train/val/test subsets,
then clip the raster into three datasets.
"""


input_tif = r''
input_shp_dir = r''
output_dir = r't'

exclude_shp = None # 可选，排除的矢量文件路径，排除区域内的像素不参与采样

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

block_size = 9
output_format = 'bin'
seed = 42
valid_sample_dir_name = 'valid_samples'
meta_file_name = '.meta.json'
split_part_names = ['shp_train', 'shp_val', 'shp_test']


def validate_split_ratios(train_ratio_value: float, val_ratio_value: float, test_ratio_value: float) -> None:
    total = float(train_ratio_value) + float(val_ratio_value) + float(test_ratio_value)
    if min(train_ratio_value, val_ratio_value, test_ratio_value) < 0:
        raise ValueError('train/val/test ratios cannot be negative.')
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f'train/val/test ratios must sum to 1.0, got {total:.6f}.')


def _feature_has_pixels_in_raster(feature: ogr.Feature, source_layer: ogr.Layer,
                                  raster_ds: gdal.Dataset,
                                  valid_mask: np.ndarray | None = None) -> bool:
    geom = feature.GetGeometryRef()
    if geom is None or geom.IsEmpty():
        return False

    geotrans = raster_ds.GetGeoTransform()
    width = raster_ds.RasterXSize
    height = raster_ds.RasterYSize
    flat_type = ogr.GT_Flatten(geom.GetGeometryType())

    if flat_type == ogr.wkbPoint:
        geo_x, geo_y = geom.GetX(), geom.GetY()
        col = int((geo_x - geotrans[0]) / geotrans[1])
        row = int((geo_y - geotrans[3]) / geotrans[5])
        if not (0 <= row < height and 0 <= col < width):
            return False
        return valid_mask is None or bool(valid_mask[row, col])

    mem_raster = gdal.GetDriverByName('MEM').Create('', width, height, 1, gdal.GDT_Byte)
    mem_raster.SetGeoTransform(geotrans)
    mem_raster.SetProjection(raster_ds.GetProjection())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(0)

    mem_vector = ogr.GetDriverByName('Memory').CreateDataSource('')
    mem_layer = mem_vector.CreateLayer(
        'feature',
        srs=source_layer.GetSpatialRef(),
        geom_type=source_layer.GetGeomType(),
    )
    mem_feature = ogr.Feature(mem_layer.GetLayerDefn())
    mem_feature.SetGeometry(geom.Clone())
    mem_layer.CreateFeature(mem_feature)
    mem_feature = None

    gdal.RasterizeLayer(mem_raster, [1], mem_layer, burn_values=[1])
    feature_mask = mem_band.ReadAsArray().astype(bool)
    if valid_mask is not None:
        if valid_mask.shape != feature_mask.shape:
            raise ValueError(f'valid_mask shape {valid_mask.shape} does not match raster shape {feature_mask.shape}')
        feature_mask = feature_mask & valid_mask
    has_pixels = bool(np.any(feature_mask))
    mem_vector = None
    mem_raster = None
    return has_pixels


def create_valid_sample_dir(source_shp_dir: str, raster_path: str, target_output_dir: str,
                            valid_mask: np.ndarray | None = None) -> str:
    valid_sample_dir = os.path.join(target_output_dir, valid_sample_dir_name)
    os.makedirs(valid_sample_dir, exist_ok=True)
    shapefile_driver = ogr.GetDriverByName('ESRI Shapefile')
    for old_shp in search_files_in_directory(valid_sample_dir, extension='.shp'):
        shapefile_driver.DeleteDataSource(old_shp)

    raster_ds = gdal.Open(raster_path)
    if raster_ds is None:
        raise RuntimeError(f'Cannot open raster: {raster_path}')

    if os.path.isdir(source_shp_dir):
        shp_files = sorted(search_files_in_directory(source_shp_dir, extension='.shp'))
    elif os.path.isfile(source_shp_dir):
        shp_files = [source_shp_dir]
    else:
        raise RuntimeError(f'Invalid shapefile path: {source_shp_dir}')

    if not shp_files:
        raise RuntimeError(f'No shapefiles found in: {source_shp_dir}')

    kept_files = 0
    kept_features = 0
    print("=======================Checking valid samples...=======================")
    print(f'Valid sample directory: {valid_sample_dir}')

    for idx, shp_path in enumerate(shp_files):
        shp_ds = ogr.Open(shp_path)
        if shp_ds is None:
            print(f'  [{idx + 1}/{len(shp_files)}] skip, cannot open: {shp_path}')
            continue

        layer = shp_ds.GetLayer()
        valid_records = []
        layer.ResetReading()
        for feature in layer:
            if _feature_has_pixels_in_raster(feature, layer, raster_ds, valid_mask=valid_mask):
                valid_records.append({'feature': feature.Clone()})

        shp_name = os.path.basename(shp_path)
        if valid_records:
            out_shp = os.path.join(valid_sample_dir, shp_name)
            _write_feature_subset_to_shp(out_shp, layer, valid_records)
            kept_files += 1
            kept_features += len(valid_records)
            print(f'  [{idx + 1}/{len(shp_files)}] keep {shp_name}: {len(valid_records)} valid feature(s)')
        else:
            print(f'  [{idx + 1}/{len(shp_files)}] drop {shp_name}: no valid feature in raster extent')
        shp_ds = None

    raster_ds = None
    if kept_files == 0:
        raise RuntimeError('No valid shapefile remains after checking raster extent.')

    print(f'Valid sample check completed: {kept_files} file(s), {kept_features} feature(s).')
    return valid_sample_dir


def _format_ratio(value: float) -> str:
    return f'{value:.0%}' if abs(value * 100 - round(value * 100)) < 1e-6 else f'{value:.2%}'


def _get_shp_feature_info(shp_path: str) -> tuple[int, str]:
    shp_ds = ogr.Open(shp_path)
    if shp_ds is None:
        return 0, 'unknown'
    layer = shp_ds.GetLayer()
    count = layer.GetFeatureCount()
    geom_type = ogr.GT_Flatten(layer.GetGeomType())
    shp_ds = None
    if geom_type in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
        return count, 'poly'
    if geom_type in (ogr.wkbPoint, ogr.wkbMultiPoint):
        return count, 'pt'
    return count, 'geom'


def build_valid_sample_mask(raster_path: str, exclude_vector_path: str | None) -> tuple[np.ndarray | None, dict]:
    if exclude_vector_path in (None, '', 'None'):
        return None, {'enabled': False, 'path': None, 'masked_pixels': 0}
    exclude_mask, _, _ = Mianvector2mask(exclude_vector_path, tif_path=raster_path, fill_value=1)
    exclude_mask = exclude_mask.astype(bool)
    valid_mask = ~exclude_mask
    masked_pixels = int(np.sum(exclude_mask))
    print(f'Exclude vector mask enabled: {exclude_vector_path}')
    print(f'  non-sampling pixels: {masked_pixels}')
    return valid_mask, {
        'enabled': True,
        'path': exclude_vector_path,
        'masked_pixels': masked_pixels,
    }


def write_crop_meta(
    target_output_dir: str,
    part_dirs: list[tuple[str, str]],
    ratios: list[float],
    random_seed: int,
    label_map: dict[str, int],
    exclude_info: dict | None = None,
    source_raster: str | None = None,
    crop_block_size: int | None = None,
    crop_output_format: str | None = None,
) -> str:
    split_stats = {}

    for part_name, part_dir in part_dirs:
        shp_files = sorted(search_files_in_directory(part_dir, extension='.shp'))
        for shp_path in shp_files:
            source_name = resolve_source_name(shp_path, part_name)
            count, geom_name = _get_shp_feature_info(shp_path)
            split_stats.setdefault(source_name, {})[part_name] = [count, geom_name]

    # 仅记录真实存在 .datasets.txt 的子集（无样本子集 clip 不会生成该文件）
    datasets = {}
    for part_name, _ in part_dirs:
        subset_name = part_name.replace('shp_', '')
        rel_path = os.path.join(f'{subset_name}_dataset', '.datasets.txt').replace('\\', '/')
        if os.path.isfile(os.path.join(target_output_dir, rel_path)):
            datasets[subset_name] = rel_path

    out_classes = (max(label_map.values()) + 1) if label_map else None

    meta = {
        'out_dir': os.path.abspath(target_output_dir),
        'ratios': {name: ratio for (name, _), ratio in zip(part_dirs, ratios)},
        'seed': int(random_seed),
        'exclude_mask': {
            'enabled': bool((exclude_info or {}).get('enabled', False)),
            'path': (exclude_info or {}).get('path'),
            'masked_pixels': int((exclude_info or {}).get('masked_pixels', 0)),
        },
        'source_raster': source_raster,
        'block_size': crop_block_size,
        'output_format': crop_output_format,
        'out_classes': out_classes,
        'label_mapping': {name: label_map[name] for name in sorted(label_map)},
        'split_feature_counts': {name: split_stats[name] for name in sorted(split_stats)},
        'datasets': datasets,
    }

    meta_path = os.path.join(target_output_dir, meta_file_name)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'Meta file saved to: {meta_path}')
    return meta_path


def split_shp_dir_to_train_val_test(
    source_shp_dir: str,
    target_output_dir: str,
    train_ratio_value: float,
    val_ratio_value: float,
    test_ratio_value: float,
    random_seed: int,
    valid_mask: np.ndarray | None = None,
    valid_mask_tif: str | gdal.Dataset | None = None,
) -> tuple[str, str, str]:
    part_dirs = batch_random_split_shp(
        input_shp_dir=source_shp_dir,
        output_dir=target_output_dir,
        ratios=[train_ratio_value, val_ratio_value, test_ratio_value],
        seed=random_seed,
        part_names=split_part_names,
        valid_mask=valid_mask,
        valid_mask_tif=valid_mask_tif,
    )
    return part_dirs[0], part_dirs[1], part_dirs[2]


def main() -> None:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    valid_mask, exclude_info = build_valid_sample_mask(input_tif, exclude_shp)
    valid_shp_dir = create_valid_sample_dir(
        source_shp_dir=input_shp_dir,
        raster_path=input_tif,
        target_output_dir=output_dir,
        valid_mask=valid_mask,
    )

    # 在划分前的全集上构建一次 0-based 连续标签映射，保证三个子集标签一致
    valid_shp_files = sorted(search_files_in_directory(valid_shp_dir, extension='.shp'))
    label_map = build_label_map_from_shps(valid_shp_files)

    train_shp_dir, val_shp_dir, test_shp_dir = split_shp_dir_to_train_val_test(
        source_shp_dir=valid_shp_dir,
        target_output_dir=output_dir,
        train_ratio_value=train_ratio,
        val_ratio_value=val_ratio,
        test_ratio_value=test_ratio,
        random_seed=seed,
        valid_mask=valid_mask,
        valid_mask_tif=input_tif,
    )
    print("=======================Cropping datasets...=======================")
    split_dirs = [
        ('shp_train', train_shp_dir),
        ('shp_val', val_shp_dir),
        ('shp_test', test_shp_dir),
    ]
    for part_name, subset_shp_dir in split_dirs:
        subset_name = part_name.replace('shp_', '')
        subset_output_dir = os.path.join(output_dir, f'{subset_name}_dataset')
        clip_by_multishp(
            subset_output_dir,
            input_tif,
            subset_shp_dir,
            block_size=block_size,
            out_tif_name=subset_name,
            output_format=output_format,
            valid_mask=valid_mask,
            label_map=label_map,
            part_name=part_name,
        )
        print(f'{subset_name.capitalize()} dataset clipping completed.')

    # 裁剪完成后再写 meta，使 datasets 字段能反映实际生成的 .datasets.txt
    write_crop_meta(
        target_output_dir=output_dir,
        part_dirs=split_dirs,
        ratios=[train_ratio, val_ratio, test_ratio],
        random_seed=seed,
        label_map=label_map,
        exclude_info=exclude_info,
        source_raster=input_tif,
        crop_block_size=block_size,
        crop_output_format=output_format,
    )


if __name__ == '__main__':
    main()
