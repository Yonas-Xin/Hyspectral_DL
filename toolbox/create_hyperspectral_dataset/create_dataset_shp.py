import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osgeo import gdal, ogr, osr

"""
一键生成空白样本 SHP 文件，供 split_and_clip_dataset 使用。

本脚本以参考影像的坐标范围和投影为基准，批量创建指定数量的空白
Shapefile（要素类型可选：点 / 面），命名规则为：
    Label_{label}_{字段名}_{日期}.shp

其中 label 从 0 开始递增，字段名由 field_names 列表指定。
如果 num_shp 大于 field_names 的长度，多余的文件使用 unknown 作为字段名。

生成后将 output_dir 目录作为 split_and_clip_dataset.py 中的 input_shp_dir
即可进行训练/验证/测试集划分和裁剪。
"""

# 几何类型字符串 -> OGR 常量
_GEOM_TYPE_MAP = {
    'point': ogr.wkbPoint,
    'polygon': ogr.wkbPolygon,
}


def _get_raster_info(tif_path: str) -> tuple[tuple, str]:
    """打开影像并返回 (geotransform, projection)。"""
    ds = gdal.Open(tif_path)
    if ds is None:
        raise RuntimeError(f'无法打开参考影像: {tif_path}')
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None
    return geotransform, projection


def create_empty_shp(output_path: str, projection_wkt: str, geom_type: str = 'point') -> bool:
    """
    创建一个带有 'class' 整型字段的空白 Shapefile。

    参数:
        output_path (str): 输出 .shp 文件完整路径
        projection_wkt (str): 投影坐标系的 WKT 字符串
        geom_type (str): 几何类型，'point'（点）或 'polygon'（面）
    """
    geom_key = geom_type.strip().lower()
    if geom_key not in _GEOM_TYPE_MAP:
        raise ValueError(f"不支持的 geom_type: {geom_type}（仅支持 {list(_GEOM_TYPE_MAP.keys())}）")
    ogr_geom_type = _GEOM_TYPE_MAP[geom_key]

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if driver is None:
        raise RuntimeError('ESRI Shapefile 驱动不可用')

    # 如果已存在则跳过，不覆盖已标注的样本
    if os.path.exists(output_path):
        return False  # 表示已跳过

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    data_source = driver.CreateDataSource(output_path)
    if data_source is None:
        raise RuntimeError(f'无法创建 Shapefile: {output_path}')

    srs = osr.SpatialReference()
    if projection_wkt:
        srs.ImportFromWkt(projection_wkt)

    layer_name = Path(output_path).stem
    layer = data_source.CreateLayer(layer_name, geom_type=ogr_geom_type, srs=srs)
    if layer is None:
        raise RuntimeError(f'无法在 Shapefile 中创建图层: {output_path}')

    # 添加 'class' 字段，与 point_shp_to_mask 读取逻辑保持一致
    field_defn = ogr.FieldDefn('class', ogr.OFTInteger)
    layer.CreateField(field_defn)

    data_source = None  # 刷新并关闭
    return True  # 表示已创建


def batch_create_sample_shp(
    tif_path: str,
    out_dir: str,
    count: int,
    names: list[str] | None = None,
    geom_type: str = 'point',
) -> list[str]:
    """
    批量创建空白样本 SHP 文件。

    参数:
        tif_path (str): 参考影像路径
        out_dir (str): 输出目录
        count (int): 要创建的 SHP 文件数量
        names (list[str]): 字段名列表，长度不足时补齐 'unknown'
        geom_type (str): 几何类型，'point'（点）或 'polygon'（面）

    返回:
        list[str]: 已创建的 SHP 文件路径列表
    """
    if count <= 0:
        raise ValueError('num_shp 必须大于 0')

    _, projection = _get_raster_info(tif_path)

    # 对齐字段名列表长度
    if names is None:
        names = []
    resolved_names: list[str] = []
    for i in range(count):
        if i < len(names):
            resolved_names.append(str(names[i]))
        else:
            resolved_names.append('unknown')

    date_str = datetime.now().strftime('%Y%m%d')
    created: list[str] = []
    skipped: list[str] = []

    os.makedirs(out_dir, exist_ok=True)

    for label, field_name in enumerate(resolved_names):
        # 命名规则：Label_{label}_{字段名}_{日期}.shp
        # 按文件名排序后，label 索引与 clip_by_multishp 中的 idx 一致
        filename = f'Label_{label}_{field_name}_{date_str}.shp'
        output_path = os.path.join(out_dir, filename)
        ok = create_empty_shp(output_path, projection, geom_type=geom_type)
        if ok:
            created.append(output_path)
            print(f'  [√] 已创建: {filename}')
        else:
            skipped.append(output_path)
            print(f'  [~] 已跳过（文件已存在）: {filename}')

    return created, skipped


def main() -> None:
    print('=' * 60)
    print('批量创建样本 SHP 文件')
    print('=' * 60)
    print(f'参考影像  : {input_tif}')
    print(f'输出目录  : {output_dir}')
    print(f'创建数量  : {num_shp}')
    print(f'几何类型  : {geom_type}')
    print(f'字段名列表: {field_names}')
    print('-' * 60)

    created_files, skipped_files = batch_create_sample_shp(
        tif_path=input_tif,
        out_dir=output_dir,
        count=num_shp,
        names=field_names,
        geom_type=geom_type,
    )

    print('-' * 60)
    print(f'完成！新建 {len(created_files)} 个，跳过 {len(skipped_files)} 个（已存在）。')
    if created_files:
        print('请使用 GIS 软件（如 QGIS/ArcGIS）在各 SHP 文件中标注样本点，')
        print('然后将该目录作为 split_and_clip_dataset.py 中的 input_shp_dir 使用。')

# ============================= 用户配置区 =============================

# 参考影像路径（用于读取投影和坐标范围）
input_tif = r''

# 输出文件夹（SHP 文件保存位置）
output_dir = r''

# 要创建的 SHP 文件数量（每个代表一类）
num_shp = 15

# 几何类型：'point' 表示点要素，'polygon' 表示面要素
geom_type = 'polygon'

# 字段名列表（可选，与类别一一对应）
# 若长度不足，多余类别使用 'unknown' 代替
field_names = []

# ============================= 用户配置区结束 ========================
if __name__ == '__main__':
    main()
