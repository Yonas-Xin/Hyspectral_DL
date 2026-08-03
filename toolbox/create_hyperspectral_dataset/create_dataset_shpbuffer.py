"""
矢量要素缓冲区批量处理工具

功能：
  1. 输入一个文件夹，批量处理其中所有 .shp 文件
  2. 对每个 shp 中的要素建立指定距离的缓冲区
  3. 如果缓冲区重叠，则合并为一个要素（UnaryUnion + 拆分多部件）
  4. 输出无拓扑错误的结果 shp

使用方式：
  修改下方配置参数后运行：
  python toolbox/buffer_and_merge.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osgeo import ogr, osr

# ========================= 配置参数 =========================

input_shp_dir = r''       # 输入 shp 文件夹路径
output_shp_dir = r''      # 输出 shp 文件夹路径
buffer_distance = 60.0    # 缓冲区距离（单位与 shp 坐标系一致，如米或度）
quad_segs = 30            # 缓冲区圆弧近似的分段数（越大越圆滑）

# ========================= 逻辑 =========================

ogr.UseExceptions()


def buffer_and_merge_shp(input_shp_path: str, output_shp_path: str,
                         dist: float, segments: int = 30) -> int:
    """
    对单个 shp 文件中的所有要素建立缓冲区，重叠部分合并。

    Args:
        input_shp_path: 输入 shp 路径
        output_shp_path: 输出 shp 路径
        dist: 缓冲区距离
        segments: 圆弧近似分段数

    Returns:
        输出要素数量
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # 读取输入
    in_ds = ogr.Open(input_shp_path, 0)
    if in_ds is None:
        raise RuntimeError(f"Cannot open: {input_shp_path}")
    in_layer = in_ds.GetLayer()
    srs = in_layer.GetSpatialRef()
    in_geom_type = in_layer.GetGeomType()

    # 第一步：对所有要素建立缓冲区
    buffer_geoms = []
    for feature in in_layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        # MakeValid 修复可能存在的拓扑错误
        if not geom.IsValid():
            geom = geom.MakeValid()
        buf = geom.Buffer(dist, segments)
        if buf is not None and not buf.IsEmpty():
            buffer_geoms.append(buf)

    in_layer.ResetReading()

    if not buffer_geoms:
        print(f"  WARNING: No valid geometries in {input_shp_path}")
        in_ds = None
        return 0

    # 第二步：合并所有重叠的缓冲区（UnaryUnion）
    merged = buffer_geoms[0].Clone()
    for g in buffer_geoms[1:]:
        merged = merged.Union(g)

    # MakeValid 确保合并后无拓扑错误
    if not merged.IsValid():
        merged = merged.MakeValid()

    # 第三步：写出结果
    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)
    if os.path.exists(output_shp_path):
        driver.DeleteDataSource(output_shp_path)

    out_ds = driver.CreateDataSource(output_shp_path)
    out_layer = out_ds.CreateLayer(
        'buffered', srs=srs, geom_type=ogr.wkbPolygon
    )

    # 复制原始字段定义（可选）
    in_layer_defn = in_layer.GetLayerDefn()
    for i in range(in_layer_defn.GetFieldCount()):
        out_layer.CreateField(in_layer_defn.GetFieldDefn(i))

    # 如果合并后是 MultiPolygon / GeometryCollection，拆分为单个 Polygon
    out_count = 0
    geom_type = merged.GetGeometryType()

    if geom_type in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        # 单个多边形
        feat = ogr.Feature(out_layer.GetLayerDefn())
        feat.SetGeometry(merged)
        out_layer.CreateFeature(feat)
        out_count = 1
    elif geom_type in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D,
                       ogr.wkbGeometryCollection, ogr.wkbGeometryCollection25D):
        # 多部件 → 逐个拆出
        for i in range(merged.GetGeometryCount()):
            sub = merged.GetGeometryRef(i)
            if sub is None or sub.IsEmpty():
                continue
            # 只保留 Polygon 类型
            sub_type = sub.GetGeometryType()
            if sub_type not in (ogr.wkbPolygon, ogr.wkbPolygon25D):
                continue
            feat = ogr.Feature(out_layer.GetLayerDefn())
            feat.SetGeometry(sub)
            out_layer.CreateFeature(feat)
            out_count += 1
    else:
        print(f"  WARNING: Unexpected geometry type after union: {ogr.GeometryTypeToName(geom_type)}")

    out_ds = None
    in_ds = None
    return out_count


def batch_buffer_and_merge(in_dir: str, out_dir: str,
                           dist: float, segments: int = 30) -> None:
    """
    批量处理文件夹中所有 .shp 文件。

    Args:
        in_dir: 输入文件夹
        out_dir: 输出文件夹
        dist: 缓冲区距离
        segments: 圆弧近似分段数
    """
    os.makedirs(out_dir, exist_ok=True)

    shp_files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith('.shp')])
    if not shp_files:
        print(f"No .shp files found in: {in_dir}")
        return

    print(f"Found {len(shp_files)} shp file(s) in: {in_dir}")
    print(f"Buffer distance: {dist}, Quad segments: {segments}")
    print("=" * 60)

    for shp_name in shp_files:
        in_path = os.path.join(in_dir, shp_name)
        out_path = os.path.join(out_dir, shp_name)
        print(f"Processing: {shp_name} ...", end=" ")
        try:
            n = buffer_and_merge_shp(in_path, out_path, dist=dist, segments=segments)
            print(f"→ {n} feature(s)")
        except Exception as e:
            print(f"ERROR: {e}")

    print("=" * 60)
    print("All done!")


def main():
    if not input_shp_dir or not output_shp_dir:
        print("ERROR: Please set input_shp_dir and output_shp_dir in the script.")
        return
    batch_buffer_and_merge(input_shp_dir, output_shp_dir,
                           dist=buffer_distance, segments=quad_segs)


if __name__ == '__main__':
    main()
