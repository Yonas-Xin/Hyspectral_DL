import os.path
import json
import re
from collections import defaultdict
try:
    from osgeo import gdal,ogr,osr
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import random
from utils import search_files_in_directory, write_list_to_txt, get_full_academic_color_256, hex_to_rgb

GDAL2NP_TYPE = { # GDAL数据类型与numpy数据类型的映射
    gdal.GDT_Byte: ('uint8', np.uint8),
    gdal.GDT_UInt16: ('uint16', np.uint16),
    gdal.GDT_Int16: ('int16', np.int16),
    gdal.GDT_UInt32: ('uint32', np.uint32),
    gdal.GDT_Int32: ('int32', np.int32),
    gdal.GDT_Float32: ('float32', np.float32),
    gdal.GDT_Float64: ('float64', np.float64)
}
NP2GDAL_TYPE = {
    np.dtype('uint8'): gdal.GDT_Byte,
    np.dtype('uint16'): gdal.GDT_UInt16,
    np.dtype('int16'): gdal.GDT_Int16,
    np.dtype('uint32'): gdal.GDT_UInt32,
    np.dtype('int32'): gdal.GDT_Int32,
    np.dtype('float32'): gdal.GDT_Float32,
    np.dtype('float64'): gdal.GDT_Float64
}

# GDAL 数据类型 -> ENVI 数据类型编号映射
_GDAL_TO_ENVI_DTYPE = {
    gdal.GDT_Byte:      1,
    gdal.GDT_Int16:     2,
    gdal.GDT_Int32:     3,
    gdal.GDT_Float32:   4,
    gdal.GDT_Float64:   5,
    gdal.GDT_UInt16:    12,
    gdal.GDT_UInt32:    13,
    gdal.GDT_CFloat32:  6,
    gdal.GDT_CFloat64:  9,
}


def _write_bin_meta_file(meta_path: str,
                         bands: int,
                         height: int,
                         width: int,
                         dtype_name: str,
                         count: int,
                         layout: str = 'BHW',
                         source_image_path: str | None = None,
                         list_path: str | None = None) -> None:
    """Write metadata for raw bin samples."""
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    meta = obj
        except Exception:
            meta = {}

    meta.update({
        'layout': layout,
        'bands': int(bands),
        'height': int(height),
        'width': int(width),
        'dtype': str(dtype_name),
        'count': int(count),
        'updated_at': datetime.now().isoformat(timespec='seconds')
    })
    if source_image_path is not None:
        meta['source_image_path'] = str(source_image_path)
    if list_path is not None:
        meta['list_path'] = str(list_path)

    out_dir = os.path.dirname(meta_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _parse_metadata_list(raw: str) -> np.ndarray:
    """将 '{v1, v2, ...}' 或 'v1, v2, ...' 形式的字符串解析为 numpy 数组"""
    s = raw.strip().strip('{}')
    return np.fromstring(s, sep=',') if s else np.array([])

def write_envi_hdr(tif_path: str, metadata: dict, dataset: gdal.Dataset) -> str:
    """
    在 TIF 文件旁边生成一个与 ENVI 格式兼容的 .hdr 头文件。
    :param tif_path:  输出的 TIF 文件路径
    :param metadata:  元数据字典，至少应包含 'wavelength' 键
    :param dataset:   输出的 GDAL Dataset（已写入数据）
    :return: 生成的 .hdr 文件路径
    """
    hdr_path = os.path.splitext(tif_path)[0] + '.hdr'
    out_dir = os.path.dirname(hdr_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    samples = dataset.RasterXSize
    lines = dataset.RasterYSize
    bands = dataset.RasterCount
    data_type = dataset.GetRasterBand(1).DataType
    envi_dtype = _GDAL_TO_ENVI_DTYPE.get(data_type, 4)  # 默认 float32
    wl_units = metadata.get('wavelength units', 'Nanometers')
    description = metadata.get('description', '')
    wavelengths = _parse_metadata_list(metadata.get('wavelength', ''))
    fwhm = _parse_metadata_list(metadata.get('fwhm', ''))
    def _format_list(arr, fmt='{:.6f}'):
        items = [fmt.format(v) for v in arr]
        lines_out = []
        for i in range(0, len(items), 5):
            chunk = items[i:i+5]
            lines_out.append(' ' + ', '.join(chunk))
        return ',\n'.join(lines_out)
    
    with open(hdr_path, 'w', encoding='utf-8') as f:
        f.write('ENVI\n')
        if description:
            f.write(f'description = {{{description}}}\n')
        f.write(f'samples = {samples}\n')
        f.write(f'lines   = {lines}\n')
        f.write(f'bands   = {bands}\n')
        f.write(f'header offset = 0\n')
        f.write(f'file type = TIFF\n')
        f.write(f'data type = {envi_dtype}\n')
        f.write(f'interleave = bip\n')
        f.write(f'byte order = 0\n')
        f.write(f'wavelength units = {wl_units}\n')
        
        if len(wavelengths) == bands:
            f.write('wavelength = {\n')
            f.write(_format_list(wavelengths))
            f.write('}\n')
        
        if len(fwhm) == bands:
            f.write('fwhm = {\n')
            f.write(_format_list(fwhm))
            f.write('}\n')
        
        band_names = []
        for i in range(1, bands + 1):
            desc = dataset.GetRasterBand(i).GetDescription()
            if desc:
                band_names.append(desc)
            else:
                band_names.append(f'Band {i}')
        f.write('band names = {\n')
        name_lines = []
        for i in range(0, len(band_names), 5):
            chunk = band_names[i:i+5]
            name_lines.append(' ' + ', '.join(chunk))
        f.write(',\n'.join(name_lines))
        f.write('}\n')
    return hdr_path

# ==========================================================================================
# 写tif文件，核心函数
# ==========================================================================================
def write_data_to_tif(output_file: str, 
                      data: np.ndarray, 
                      geotransform: tuple, 
                      projection: str, 
                      nodata_value: int | float | None = None, 
                      mask: np.ndarray | None = None,
                      metadata: dict | None = None):
    """
    将数组数据写入GeoTIFF文件
    
    参数:
        output_file (str): 输出文件路径
        data (np.ndarray): 三维数组（波段,行,列）
        geotransform (tuple): 6参数地理变换
        projection (str): WKT格式的坐标参考系统
        nodata_value (int/float): NoData值, 默认0
        mask (np.ndarray): 可选的掩码数组, 形状应与data的空间维度匹配, 用于保存有效数据区域，其余区域设为nodata_value
        metadata (dict): 可选的元数据字典, 包含如 wavelength, fwhm, wavelength units 等键值对。
                         若提供且包含 wavelength, 将同时生成 ENVI 格式 .hdr 头文件
    
    异常:
        IOError: 当文件创建失败时
    """
    # 处理二维数组情况（单波段）
    if len(data.shape) == 2:
        rows, cols = data.shape
        bands = 1
        data = data.reshape((1, rows, cols))  # 转换为三维
        if data.dtype == np.uint8:
            create_color_table = True # 如果是uint8类型则创建颜色表
        else: create_color_table = False
    elif len(data.shape) == 3:
        bands, rows, cols = data.shape
        create_color_table = False
    else:
        create_color_table = False
        raise ValueError("输入数据必须是二维或三维数组")
    
    try:
        dtype = NP2GDAL_TYPE[data.dtype]
    except KeyError:
        raise ValueError(f"不支持的数据类型: {data.dtype}. 支持的类型包括: {list(NP2GDAL_TYPE.keys())}")
    if mask is not None: # 如果提供了掩码, 则将掩码区域外的数据设为nodata_value（默认为0）
        mask = mask.astype(np.bool_)
        data = data.transpose((1, 2, 0))
        data[~mask] = nodata_value
        data = data.transpose((2, 0, 1))
    # 创建文件
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_file, cols, rows, bands, dtype,
                            options=['INTERLEAVE=PIXEL'])
    if dataset is None:
        raise IOError(f"无法创建文件 {output_file}")
    # 设置地理变换和投影
    if geotransform is not None:
        dataset.SetGeoTransform(geotransform)
    if projection is not None:
        dataset.SetProjection(projection)
    # 写入数据
    if create_color_table: # 如果需要创建颜色表
        rgb_colors = [tuple(hex_to_rgb(color)) for color in get_full_academic_color_256()]
        max_val = min(len(rgb_colors), 255)
        out_band = dataset.GetRasterBand(1)
        colors = gdal.ColorTable()
        for val in range(max_val):
            colors.SetColorEntry(int(val), rgb_colors[int(val)])
        out_band.SetRasterColorTable(colors)
        out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(data[i,:,:])
        if nodata_value is not None:
            band.SetNoDataValue(nodata_value)  # 设置 NoData 值
    # 写入元数据
    if metadata:
        for key, value in metadata.items():
            dataset.SetMetadataItem(key, str(value))
        dataset.FlushCache()
        # 如果元数据中包含 wavelength，生成 ENVI .hdr 头文件
        if 'wavelength' in metadata:
            hdr_path = write_envi_hdr(output_file, metadata, dataset)
            print(f"INFO: 已生成 ENVI 头文件: {hdr_path}")
    # 释放资源
    dataset.FlushCache()
    dataset = None
    return output_file

def subset_image(input_img: str | gdal.Dataset, output_path: str, band_indices: list[int], metadata: dict):
    """
    根据波段索引提取影像波段，保存为BIP交错的TIFF，并同步更新元数据（如波长和FWHM）。

    参数:
        input_img: str 或 gdal.Dataset, 输入影像的文件路径或已打开的 GDAL Dataset 对象。
        band_indices: list of int, 需要提取的波段索引列表（注意：GDAL 波段索引从 1 开始）。
        metadata: dict, 原始影像的元数据字典。期望包含 'wavelength' 和 'fwhm' 键，
                       值为列表或逗号分隔的字符串，代表所有原始波段的参数。
        output_path: str, 输出的 TIFF 影像保存路径。
    """
    if isinstance(input_img, str):
        if not os.path.exists(input_img):
            raise FileNotFoundError(f"找不到输入文件: {input_img}")
        ds = gdal.Open(input_img)
    elif isinstance(input_img, gdal.Dataset):
        ds = input_img
    else:
        raise TypeError("input_img 必须是文件路径字符串或 gdal.Dataset 对象。")

    if ds is None:
        raise ValueError("无法打开输入影像，请检查数据格式。")
    width = ds.RasterXSize
    height = ds.RasterYSize
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    translate_options = gdal.TranslateOptions(
        format='GTiff',
        bandList=band_indices,
        creationOptions=['INTERLEAVE=PIXEL'],
    )
    out_ds = gdal.Translate(output_path, ds, options=translate_options)

    if out_ds is None:
        raise ValueError("无法创建输出影像，请检查输出路径或权限。")

    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)

    for out_idx, in_idx in enumerate(band_indices, start=1):
        in_band = ds.GetRasterBand(in_idx)
        if in_band is None:
            raise ValueError(f"输入影像中不存在波段索引: {in_idx}")
        nodata = in_band.GetNoDataValue()
        if nodata is not None:
            out_ds.GetRasterBand(out_idx).SetNoDataValue(nodata)
    final_metadata = ds.GetMetadata().copy()
    if metadata and isinstance(metadata, dict):
        for key, val in metadata.items():
            final_metadata[key] = str(val)
    out_ds.SetMetadata(final_metadata)
    out_ds.FlushCache()
    if 'wavelength' in final_metadata:
        write_envi_hdr(output_path, final_metadata, out_ds)
    out_ds = None
    if isinstance(input_img, str):
        ds = None 

def read_tif(tif_path: str | gdal.Dataset) -> tuple[tuple, str, int, int, int]:
    """读取tif文件，返回地理变换、投影、高宽、波段数"""
    need_close = False
    if isinstance(tif_path, str):
        im_dataset = gdal.Open(tif_path)
        if im_dataset is None:
            raise RuntimeError(f"无法打开影像文件: {tif_path}")
        need_close = True
    elif isinstance(tif_path, gdal.Dataset):
        im_dataset = tif_path
    else:
        raise TypeError("tif_path必须是文件路径字符串或GDAL数据集对象")
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize # 宽
    im_height = im_dataset.RasterYSize # 高
    im_bands = im_dataset.RasterCount
    if need_close:
        im_dataset = None
    return im_geotrans, im_proj, im_width, im_height, im_bands
    

def Mianvector2mask(vector_path: str, tif_path: str | None = None, 
                    fill_value: int = 255, pixel_size: float = 1.0) -> tuple[np.ndarray, tuple, str]:
    """
    使用GDAL将矢量文件转换为numpy矩阵（mask矩阵）, 如果提供了width和height则使用，否则根据矢量范围和像素大小创建mask矩阵
    :param vector_path: 矢量文件路径（Shapefile）
    :param tif_path: 可选的参考tif文件路径，用于获取地理变换和投影信息
    :return: NumPy矩阵 (H, W)
    """
    vector_ds = ogr.Open(vector_path)
    if not vector_ds:
        raise RuntimeError(f"无法打开矢量文件 {vector_path}")
    layer = vector_ds.GetLayer()
    # 创建一个内存栅格数据集，存储矢量化数据
    mem_driver = gdal.GetDriverByName("MEM")
    if tif_path is not None:
        geotransform, projection, width, height, bands = read_tif(tif_path)
        mem_raster = mem_driver.Create("", width, height, 1, gdal.GDT_Int16)
        mem_raster.SetGeoTransform(geotransform)  # 设定仿射变换
        mem_raster.SetProjection(projection)  # 设定投影
    else: # 如果没有提供geotransform，则根据矢量范围和像素大小创建mask
        x_min, x_max, y_min, y_max = layer.GetExtent()
        x_res = int((x_max - x_min) / pixel_size)
        y_res = int((y_max - y_min) / pixel_size)
        geotransform = (x_min, pixel_size, 0, y_max, 0, -pixel_size)
        projection = layer.GetSpatialRef().ExportToWkt()
        mem_raster = mem_driver.Create("", x_res, y_res, 1, gdal.GDT_Int16)
        mem_raster.SetGeoTransform(geotransform)  # 设定仿射变换
        mem_raster.SetProjection(projection)  # 设定投影
    band = mem_raster.GetRasterBand(1)
    band.Fill(0)  # 未覆盖区域填充 0
    gdal.RasterizeLayer(mem_raster, [1], layer, burn_values = [fill_value]) # 矢量覆盖区填充1
    matrix = band.ReadAsArray()
    vector_ds = None
    mem_raster = None
    return matrix, geotransform, projection

def mask_to_point_shp(mask_matrix: np.ndarray, tif_path: str | gdal.Dataset, 
                      output_shapefile: str = "./Position_mask/test.shp") -> None:
    """
    将二维矩阵mask矩阵转化为矢量点文件
    """
    # 获取矩阵的行和列
    geotransform, projection, _, _, _ = read_tif(tif_path)
    rows, cols = mask_matrix.shape
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if not driver:
        raise RuntimeError("Shapefile driver not available")
    out_dir = os.path.dirname(output_shapefile)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    data_source = driver.CreateDataSource(output_shapefile)
    # 创建一个图层，用于存储点（Point）几何
    spatial_ref = osr.SpatialReference()
    if projection:
        spatial_ref.ImportFromWkt(projection) # 定义投影坐标系
    layer = data_source.CreateLayer('points', geom_type=ogr.wkbPoint, srs=spatial_ref)
    field = ogr.FieldDefn('class', ogr.OFTInteger)
    layer.CreateField(field)
    # 遍历矩阵，提取非零值的坐标并创建点特征
    for row in range(rows):
        for col in range(cols):
            value = mask_matrix[row, col]
            if value > 0:  # 非零值表示分类
                # 创建一个点
                geo_x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
                geo_y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
                point = ogr.Geometry(ogr.wkbPoint)
                # if write_value is None:
                #     write_value = int(value-1)
                point.AddPoint(geo_x, geo_y)
                # 创建一个要素（Feature）并设置几何和属性值
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetGeometry(point)
                feature.SetField('class', 0)  # 设置分类属性值
                layer.CreateFeature(feature)  # 将特征写入图层
                feature = None  # 清理
    # 关闭数据源，保存Shapefile
    data_source = None
    print(f"{output_shapefile} has been created successfully.")

def mask_to_multipoint_shp(mask_matrix: np.ndarray, tif_path: str | gdal.Dataset,
                           output_dir: str = "./Position_mask", output_dir_name: str | None = None) -> None:
    """
    在指定文件夹下创建一个新文件夹保存样本的矢量点文件
    """
    labels = np.unique(mask_matrix)
    if output_dir_name is None:
        output_dir_name = 'SAMPLES_DIR'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    OUTPUT_DIR = os.path.join(output_dir, output_dir_name)
    counter = 1
    while os.path.exists(OUTPUT_DIR):
        OUTPUT_DIR = os.path.join(output_dir, f"{output_dir_name}_{counter}")
        counter += 1
    os.makedirs(OUTPUT_DIR)
    for label in labels:
        if label > 0:
            input = np.zeros_like(mask_matrix, dtype=np.int16)
            time.sleep(1) # 确保间隔一秒创建一个文件
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            output_shapefile = os.path.join(OUTPUT_DIR, f"{current_time}_sample.shp")
            input[mask_matrix == label] = label
            mask_to_point_shp(input, tif_path, output_shapefile)

def point_shp_to_mask(shapefile: str, tif_path: str | gdal.Dataset, value: int | None = None) -> np.ndarray:
    """
    将矢量点文件转化为二维矩阵（mask矩阵）
    mask属性值从1开始，背景为0
    :param shapefile: 矢量点文件路径
    :param tif_path: 参考tif文件路径，用于获取地理变换和矩阵大小
    :param value: 可选的指定mask的值
    """
    geotransform, projection, cols, rows, bands = read_tif(tif_path)
    mask_matrix = np.zeros((rows, cols), dtype=int)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = ogr.Open(shapefile)
    if not data_source:
        raise RuntimeError(f"Failed to open shapefile: {shapefile}")
    layer = data_source.GetLayer()
    field_idx = layer.FindFieldIndex('class', 1)
    if field_idx != -1:
        has_field_class = True
    else: has_field_class = False
    spatial_ref = layer.GetSpatialRef()
    if spatial_ref is None:
        raise RuntimeError("No spatial reference found in shapefile")
    # 处理每个要素（点）
    for feature in layer:
        # 获取点的几何体
        geometry = feature.GetGeometryRef()
        if geometry.GetGeometryType() != ogr.wkbPoint:
            continue  # 只处理点类型的几何
        geo_x, geo_y = geometry.GetX(), geometry.GetY()
        col = int((geo_x - geotransform[0]) / geotransform[1])  # 计算列索引
        row = int((geo_y - geotransform[3]) / geotransform[5])  # 计算行索引
        if 0 <= row < rows and 0 <= col < cols:        # 确保索引在矩阵范围内
            if value is not None:
                mask_matrix[row, col] = value
            # else:mask_matrix[row, col] = feature.GetField('class') + 1  # 如果没有指定值，则使用字段“class”值
            else:
                if has_field_class:
                    class_value = feature.GetField('class')
                    mask_matrix[row, col] = class_value + 1
                else: mask_matrix[row, col] = 1 # 默认点位置取值为1
    data_source = None
    return mask_matrix

def mutipoint_shp_to_mask(shapefile_dir: str, tif_path: str | gdal.Dataset) -> np.ndarray:
    """
    将指定文件夹下的所有矢量点文件转化为二维矩阵（mask矩阵）
    """
    _, _, cols, rows, _ = read_tif(tif_path)
    mask_matrix = np.zeros((rows, cols), dtype=int)
    shapefiles = search_files_in_directory(shapefile_dir, extension='.shp')
    if not shapefiles:
        raise RuntimeError(f"No shapefiles found in directory: {shapefile_dir}")
    for idx, shapefile in enumerate(shapefiles):
        temp_mask = point_shp_to_mask(shapefile, tif_path, value=idx+1)
        # 检查是否有重叠（即 mask_matrix 和 temp_mask 在相同位置都有非零值）
        overlap = np.logical_and(mask_matrix > 0, temp_mask > 0)
        if np.any(overlap):
            raise RuntimeError(
                f"There is an overlap in the sample points, please check the file! "
            )
        mask_matrix += temp_mask
    return mask_matrix

# ==========================================================================================
# 根据多shp裁剪样本集，MHMAP工程版本
# ==========================================================================================
def clip_by_position(out_dir: str, sr_img: str | gdal.Dataset, position_list: list, patch_size: int = 30, 
                     out_tif_name: str = 'img', label=None, output_format: str = 'tif') -> list[str]:
    """
    根据掩码矩阵从影像中裁剪指定大小的图像块
    
    参数:
        out_dir (str): 输出目录路径
        sr_img (str/gdal.Dataset): 输入影像路径或已打开的GDAL数据集对象
        position_list (list): 需要裁剪的位置列表，每个位置为(y, x)元组
        patch_size (int): 裁剪块大小（像素），默认30
        out_tif_name (str): 输出文件名前缀，默认'img'
        fill_value (int/float): 边缘填充值，默认0
    
    返回:
        list: 生成的图像路径列表, 格式为["path1.tif", "path2.tif", ...]
    
    异常:
        RuntimeError: 当无法打开输入文件时
        TypeError: 当sr_img参数类型无效时
    """
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    output_format = str(output_format).lower()
    if output_format not in ('tif', 'bin'):
        raise ValueError(f"Unsupported output_format: {output_format}. Expected 'tif' or 'bin'.")
    # 计算中心偏移
    if patch_size % 2 == 0:
        left_top = patch_size // 2 - 1
        right_bottom = patch_size // 2
    else:
        left_top = right_bottom = patch_size // 2

    # 读取原始影像
    need_close = False
    if isinstance(sr_img, str):
        im_dataset = gdal.Open(sr_img)
        if im_dataset is None:
            raise RuntimeError(f"无法打开影像文件: {sr_img}")
        need_close = True
    elif isinstance(sr_img, gdal.Dataset):
        im_dataset = sr_img
    else:
        raise TypeError("sr_img必须是文件路径字符串或GDAL数据集对象")
    
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    im_bands = im_dataset.RasterCount

    band = im_dataset.GetRasterBand(1)
    data_type = band.DataType # 获取数据类型
    dtype_name, numpy_dtype = GDAL2NP_TYPE.get(data_type, ('unknown', None)) # 确定numpy数据类型
    if dtype_name == 'unknown':
        raise ValueError(f"不支持的GDAL数据类型: {data_type}")
    count = len(position_list)

    pbar = tqdm(total=count) # 进度条
    idx = 0
    out_dataset = []
    for y, x in position_list:
        # 计算裁剪窗口
        x_start = x - left_top
        y_start = y - left_top
        x_end = x + right_bottom + 1
        y_end = y + right_bottom + 1
        
        # 计算实际可读取范围
        read_x = max(0, x_start)
        read_y = max(0, y_start)
        read_width = min(x_end, im_width) - read_x
        read_height = min(y_end, im_height) - read_y
        
        # 如果有有效区域可读取
        if read_width > 0 and read_height > 0: # 在影像范围内才进行裁剪
            if im_bands > 1:# 创建填充数组, 默认填充值为0
                full_data = np.full((im_bands, patch_size, patch_size), 0, dtype=numpy_dtype)
            else:
                full_data = np.full((patch_size, patch_size), 0, dtype=numpy_dtype)
            if read_width > 0 and read_height > 0:  
                # 读取实际数据
                if im_bands > 1:
                    data = im_dataset.ReadAsArray(read_x, read_y, read_width, read_height)
                    # 计算在填充数组中的位置
                    offset_x = read_x - x_start
                    offset_y = read_y - y_start
                    # 将数据放入填充数组
                    full_data[:, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
                else:
                    data = im_dataset.GetRasterBand(1).ReadAsArray(read_x, read_y, read_width, read_height)
                    offset_x = read_x - x_start
                    offset_y = read_y - y_start
                    full_data[offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
        
            # 计算新的地理变换
            new_geotrans = list(im_geotrans)
            new_geotrans[0] = im_geotrans[0] + x_start * im_geotrans[1]
            new_geotrans[3] = im_geotrans[3] + y_start * im_geotrans[5]
            
            # 保存结果
            idx += 1
            if output_format == 'tif':
                out_path = os.path.join(out_dir, f"{out_tif_name}_{idx}.tif")
                write_data_to_tif(out_path, full_data, new_geotrans, im_proj)
            else:
                out_path = os.path.join(out_dir, f"{out_tif_name}_{idx}.bin")
                if im_bands > 1:
                    full_data.tofile(out_path)  # [bands, H, W]
                else:
                    full_data[np.newaxis, :, :].tofile(out_path)  # [1, H, W]
            if label is not None:
                out_dataset.append(f"{out_path} {label}")
            else:
                # 无标签样本：仅记录样本地址
                out_dataset.append(f"{out_path}")
        pbar.update(1)
    pbar.close()
    if output_format == 'bin' and out_dir:
        try:
            bin_count = len([n for n in os.listdir(out_dir) if n.lower().endswith('.bin')])
        except Exception:
            bin_count = len(out_dataset)
        source_path = sr_img if isinstance(sr_img, str) else None
        meta_path = os.path.join(out_dir, 'raw_bip_meta.json')
        _write_bin_meta_file(
            meta_path=meta_path,
            bands=im_bands,
            height=patch_size,
            width=patch_size,
            dtype_name=dtype_name,
            count=bin_count,
            layout='BHW',
            source_image_path=source_path
        )
    if need_close:
        im_dataset = None
    return out_dataset

def clip_by_shp(out_dir: str, sr_img: str | gdal.Dataset, shp_path: str, patch_size: int = 30, 
                out_tif_name: str = 'img', label: int | None = None, 
                sample_num: int = 50000, output_format: str = 'tif',
                valid_mask: np.ndarray | None = None) -> list[str]:
    """
    根据点Shapefile从影像中裁剪指定大小的图像块
    
    参数:
        out_dir (str): 输出目录路径
        sr_img (str/gdal.Dataset): 输入影像路径或已打开的GDAL数据集对象
        shp_path (str): 点、面要素Shapefile路径
        patch_size (int): 裁剪块大小（像素），默认30
        out_tif_name (str): 输出文件名前缀，默认'img'
        fill_value (int/float): 边缘填充值，默认0
        label (int): 为输出文件名添加的标签值, 默认None
        sample_num (int): 如果是面矢量，最大采样点数量, 默认50000
        output_format (str): 输出格式，可选 'tif' 或 'bin'，默认 'tif'

    返回:
        list: 生成的图像路径列表, 格式为["path1.tif label1", "path2.tif label2", ...]
    
    异常:
        RuntimeError: 当无法打开输入文件时
        TypeError: 当sr_img参数类型无效时
    """
    # 读取原始影像
    if isinstance(sr_img, str):
        im_dataset = gdal.Open(sr_img)
        if im_dataset is None:
            raise RuntimeError(f"无法打开影像文件: {sr_img}")
    elif isinstance(sr_img, gdal.Dataset):
        im_dataset = sr_img
    else:
        raise TypeError("sr_img必须是文件路径字符串或GDAL数据集对象")
    
    im_geotrans = im_dataset.GetGeoTransform()
    if im_geotrans == (0, 1, 0, 0, 0, 1):
        print("警告: 该影像无法获取正确的地理参考信息，裁剪结果可能不准确。\n" \
        "如果是未经几何校正的dat影像，请将其几何校正或者转化为tif格式并指定地理参考后再进行裁剪。")

    band = im_dataset.GetRasterBand(1)
    data_type = band.DataType # 获取数据类型
    dtype_name, numpy_dtype = GDAL2NP_TYPE.get(data_type, ('unknown', None)) # 确定numpy数据类型
    if dtype_name == 'unknown':
        raise ValueError(f"不支持的GDAL数据类型: {data_type}")
    
    # 读取样本点
    shp_dataset = ogr.Open(shp_path)
    if shp_dataset is None:
        raise RuntimeError(f"无法打开矢量文件: {shp_path}")
    
    layer = shp_dataset.GetLayer()
    geom_type = layer.GetGeomType()
    geom_type_name = ogr.GeometryTypeToName(geom_type)
    # 判断是点矢量还是面矢量
    if geom_type_name == "Polygon": 
        mask,_,_ = Mianvector2mask(shp_path, im_dataset, fill_value=1)
        if valid_mask is not None:
            if valid_mask.shape != mask.shape:
                raise ValueError(f"valid_mask shape {valid_mask.shape} does not match image mask shape {mask.shape}")
            mask = mask & valid_mask.astype(bool)
        positions = np.column_stack(np.where(mask == 1)).tolist()
        if len(positions) > sample_num: # 如果是面矢量且采样量多大，控制采样量为sample_num
            random.seed(42)  # For reproducibility
            positions = random.choices(positions, sample_num)
    else:
        positions = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            geoX, geoY = geom.GetX(), geom.GetY()
                    # 转换坐标到像素位置
            x = int((geoX - im_geotrans[0]) / im_geotrans[1])
            y = int((geoY - im_geotrans[3]) / im_geotrans[5])
            positions.append([y, x]) # 注意这里是行列顺序
    if valid_mask is not None and geom_type_name != "Polygon":
        expected_shape = (im_dataset.RasterYSize, im_dataset.RasterXSize)
        if valid_mask.shape != expected_shape:
            raise ValueError(f"valid_mask shape {valid_mask.shape} does not match image shape {expected_shape}")
        positions = [
            [y, x] for y, x in positions
            if 0 <= y < valid_mask.shape[0] and 0 <= x < valid_mask.shape[1] and valid_mask[y, x]
        ]
    out_dataset = clip_by_position(out_dir, sr_img, positions, patch_size, out_tif_name=out_tif_name, label=label, output_format=output_format)
    im_dataset = None
    return out_dataset

# 文件名中的 Label_{N} 模式（不区分大小写），如 "Label_2_变质岩_20260324" → 2
LABEL_RE = re.compile(r'(?:^|[_\-\s])label[_\-\s](\d+)', re.IGNORECASE)


def resolve_source_name(shp_path: str, part_name: str | None = None) -> str:
    """取 shapefile 文件名（不含扩展名）作为源类别名。

    若提供 part_name 且文件名以 "_{part_name}" 结尾（划分后产生的子集后缀，
    如 "Label_2_xxx_shp_train"），则剥离该后缀以还原源类别名。
    """
    stem = os.path.splitext(os.path.basename(shp_path))[0]
    if part_name:
        suffix = f'_{part_name}'
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def build_label_map_from_shps(shp_files: list[str], part_name: str | None = None) -> dict[str, int]:
    """在一组 shapefile 上构建 {源类别名: 0-based连续标签} 的规范映射。

    规则:
        1. 优先解析文件名中的 Label_{N} 数字；按所有出现的 N 升序压缩为
           0-based 连续标签（如 Label_2,5,7 → 0,1,2）；同一 N 视为同一类别。
        2. 不含 Label_{N} 的文件作为各自独立的新类别，排在规范类别之后，
           按源类别名排序依次继续编号。
    """
    raw_by_source: dict[str, int | None] = {}
    for shp_path in shp_files:
        source_name = resolve_source_name(shp_path, part_name)
        if source_name in raw_by_source:
            continue
        match = LABEL_RE.search(os.path.splitext(os.path.basename(shp_path))[0])
        raw_by_source[source_name] = int(match.group(1)) if match else None

    # 规范类别：按 Label_{N} 升序压缩为 0-based 连续
    conforming_raws = sorted({raw for raw in raw_by_source.values() if raw is not None})
    rank_of_raw = {raw: idx for idx, raw in enumerate(conforming_raws)}
    next_rank = len(conforming_raws)

    label_map: dict[str, int] = {}
    # 非规范类别：按源名排序，接在规范类别之后继续编号
    unlabeled_sources = sorted(src for src, raw in raw_by_source.items() if raw is None)
    rank_of_unlabeled = {src: next_rank + i for i, src in enumerate(unlabeled_sources)}

    for source_name, raw in raw_by_source.items():
        label_map[source_name] = rank_of_raw[raw] if raw is not None else rank_of_unlabeled[source_name]
    return label_map


def clip_by_multishp(out_dir: str, sr_img: str | gdal.Dataset, shp_dir: str, block_size: int = 30, out_tif_name: str = 'img',
                     output_format: str = 'tif', valid_mask: np.ndarray | None = None,
                     label_map: dict[str, int] | None = None, part_name: str | None = None) -> None:
    """
    批量处理目录下多个Shapefile的裁剪任务, 并自动生成记录样本块与标签的数据集

    标签解析规则（目录模式）:
        标签统一通过 build_label_map_from_shps 归一化为 0-based 连续整数：
        优先按文件名中的 Label_{N} 数字排序压缩（如 Label_2,5,7 → 0,1,2），
        无 Label_{N} 的文件作为独立类别排在其后、按文件名继续编号。

    参数:
        out_dir (str): 输出目录路径
        sr_img (str/gdal.Dataset): 输入影像路径或已打开的GDAL数据集对象
        shp_dir (str): 包含点Shapefiles的目录路径或者是一个Shapefile文件路径
        block_size (int): 裁剪块大小（像素）, 默认30
        out_tif_name (str): 输出文件名前缀, 默认'img'
        output_format (str): 输出格式，'tif' 或 'bin'，默认'tif'
        label_map (dict[str, int]): 可选的 {源文件名: 0-based标签} 映射。
            提供时跨多次调用（train/val/test）保持同一类别同一标签；
            未提供时按当前目录内的文件自动构建一份归一化映射。
        part_name (str): 可选的子集后缀名（如 'shp_train'），用于从划分后的文件名
            还原源类别名以匹配 label_map。

    返回:
        None

    异常:
        RuntimeError: 当目录中没有Shapefile或裁剪失败时
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.isdir(shp_dir):
        point_shp_files = sorted(search_files_in_directory(shp_dir, extension='.shp'))
        if not point_shp_files:
            raise RuntimeError(f'Not shapefiles found in directory: {shp_dir}')
        if label_map is None:
            label_map = build_label_map_from_shps(point_shp_files, part_name=part_name)
        all_out_datasets = []
        for idx, point_shp in enumerate(point_shp_files):
            source_name = resolve_source_name(point_shp, part_name)
            if source_name not in label_map:
                raise RuntimeError(f'No label mapping for class "{source_name}" (file: {point_shp})')
            label = label_map[source_name]
            shp_name = os.path.basename(point_shp)
            print(f'  [{idx + 1}/{len(point_shp_files)}] {shp_name}  →  label={label}  (from canonical-map)')
            out_dataset = clip_by_shp(out_dir, sr_img, point_shp, block_size,
                                      out_tif_name=f'{out_tif_name}_label{label}',
                                      label=label, output_format=output_format,
                                      valid_mask=valid_mask)
            all_out_datasets.extend(out_dataset)
        if all_out_datasets:
            dataset_path = os.path.join(out_dir, '.datasets.txt')
            write_list_to_txt(all_out_datasets, dataset_path)
            print(f'dataset file saved to: {dataset_path}')
    elif os.path.isfile(shp_dir):
        print('single shape file: clipped as no-label samples (.txt records path only)')
        out_dataset = clip_by_shp(out_dir, sr_img, shp_dir, block_size,
                                  out_tif_name=out_tif_name, output_format=output_format,
                                  label=None, valid_mask=valid_mask)
        dataset_path = os.path.join(out_dir, '.datasets.txt')
        write_list_to_txt(out_dataset, dataset_path)
        print(f'dataset file saved to: {dataset_path}')
    else:
        raise RuntimeError(f'Invalid point_shp_dir: {shp_dir}, it should be a directory or a shapefile path')
    
# ==========================================================================================
# 栅格转矢量
# ==========================================================================================
def batch_raster_to_vector(tif_dir: str, shp_img_path: str, extension: str = '.tif', dict: dict | None = None, 
                           delete_value: int = 0, if_smooth: bool = False):
    """
    批量栅格转矢量, code by why
    :param tif_dir: 输入的需要处理的栅格文件夹
    :param shp_img_path: 输出的矢量路径
    :param extension: 栅格后缀
    :param dict: 类型字典, 如{1: "变质岩", 2: "沉积岩", ...}
    :param delete_value: 需要删除的背景值, 默认为0
    :param if_smooth: 是否平滑矢量
    :return:
    """
    def smoothing(inShp, fname, bdistance=0.001):
        """
        :param inShp: 输入的矢量路径
        :param fname: 输出的矢量路径
        :param bdistance: 缓冲区距离
        :return:
        """
        ogr.UseExceptions()
        in_ds = ogr.Open(inShp)
        in_lyr = in_ds.GetLayer()
        # 创建输出Buffer文件
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(fname):
            driver.DeleteDataSource(fname)
        # 新建DataSource, Layer
        out_ds = driver.CreateDataSource(fname)
        out_lyr = out_ds.CreateLayer(fname, in_lyr.GetSpatialRef(), ogr.wkbPolygon)
        def_feature = out_lyr.GetLayerDefn()
        # 遍历原始的Shapefile文件给每个Geometry做Buffer操作
        for feature in in_lyr:
            geometry = feature.GetGeometryRef()
            buffer = geometry.Buffer(bdistance).Buffer(-bdistance)
            out_feature = ogr.Feature(def_feature)
            out_feature.SetGeometry(buffer)
            out_lyr.CreateFeature(out_feature)
            out_feature = None
        out_ds.FlushCache()
        del in_ds, out_ds
    def raster2poly(raster, outshp, geology_dict=dict):
        """栅格转矢量
        Args:
            raster: 栅格文件名
            outshp: 输出矢量文件名
            geology_dict: 地质类型字典, 如{1: "变质岩", 2: "沉积岩", ...}
        """
        inraster = gdal.Open(raster)  # 读取路径中的栅格数据
        inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段
        prj = osr.SpatialReference()
        prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息

        drv = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outshp):  # 若文件已经存在, 则删除它继续重新做一遍
            drv.DeleteDataSource(outshp)
        Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
        Poly_layer = Polygon.CreateLayer(
            raster[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)

        newField = ogr.FieldDefn('pValue', ogr.OFTReal)
        Poly_layer.CreateField(newField)

        if geology_dict is not None:
            dzField = ogr.FieldDefn('dz', ogr.OFTString)
            dzField.SetWidth(50)  # 设置字段宽度
            Poly_layer.CreateField(dzField)

        gdal.FPolygonize(inband, None, Poly_layer, 0)  # 核心函数, 执行栅格转矢量操作
        if geology_dict is not None:
            for feature in Poly_layer:
                pvalue = feature.GetField('pValue')
                if pvalue in geology_dict:
                    feature.SetField('dz', geology_dict[pvalue])
                    Poly_layer.SetFeature(feature)
        
        Polygon.SyncToDisk()
        Polygon = None
    if '.shp' in shp_img_path:
        shp_img_path = shp_img_path[:-4]
    if shp_img_path and not os.path.exists(shp_img_path):
        os.makedirs(shp_img_path, exist_ok=True)
    if os.path.isdir(tif_dir):
        listpic = search_files_in_directory(tif_dir, extension)
    else:
        listpic = [tif_dir]
        tif_dir = os.path.dirname(tif_dir)
    for img in tqdm(listpic):
        tif_img_full_path = img
        base_name = os.path.basename(img)
        shp_full_path = shp_img_path + '/' + base_name[:-4] + '.shp'

        raster2poly(tif_img_full_path, shp_full_path, dict)

        ogr.RegisterAll()  # 注册所有的驱动

        driver = ogr.GetDriverByName('ESRI Shapefile')
        shp_dataset = ogr.Open(shp_full_path, 1)  # 0只读模式, 1读写模式
        if shp_full_path is None:
            print('Failed to open shp_1')

        ly = shp_dataset.GetLayer()

        '''删除矢量化结果中为背景的要素'''
        feature = ly.GetNextFeature()
        while feature is not None:
            gridcode = feature.GetField('pValue')
            if gridcode == delete_value:
                delID = feature.GetFID()
                ly.DeleteFeature(int(delID))
            feature = ly.GetNextFeature()
        ly.ResetReading()  # 重置
        del shp_dataset
        '''平滑矢量'''
        if if_smooth:
            smooth_shp_full_path = shp_img_path + '/' + 'smooth_' + base_name[:-4] + '.shp'
            smoothing(shp_full_path, smooth_shp_full_path, bdistance=0.15)

def mask2poly(mask_array: np.ndarray, geotransform: tuple, projection: str, outshp: str, 
              field_name: str = 'value', remove_zero: bool = True):
    """
    将内mask数组转换为面矢量
    
    Args:
        mask_array: 二维numpy数组表示的mask
        geotransform: 地理变换参数 (x_min, pixel_width, 0, y_max, 0, -pixel_height)
        projection: 投影信息 (WKT格式)
        outshp: 输出矢量文件名
        field_name: 属性字段名称
        remove_zero: 是否删除值为0的要素
    """
    driver = gdal.GetDriverByName('MEM')
    rows, cols = mask_array.shape
    
    mem_raster = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    mem_raster.SetGeoTransform(geotransform)
    mem_raster.SetProjection(projection)

    band = mem_raster.GetRasterBand(1)
    band.WriteArray(mask_array)
    band.SetNoDataValue(0)
    band.FlushCache()

    prj = osr.SpatialReference()
    prj.ImportFromWkt(projection)

    drv = ogr.GetDriverByName("ESRI Shapefile")
    out_dir = os.path.dirname(outshp)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(outshp):
        drv.DeleteDataSource(outshp)
    
    polygon_ds = drv.CreateDataSource(outshp)
    poly_layer = polygon_ds.CreateLayer(
        os.path.basename(outshp)[:-4],  # 图层名（去掉.shp后缀）
        srs=prj, 
        geom_type=ogr.wkbMultiPolygon
    )

    field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
    poly_layer.CreateField(field_defn)

    gdal.Polygonize(band, None, poly_layer, 0, [], callback=None)

    poly_layer.SyncToDisk()
    polygon_ds = None

    if remove_zero:
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shp_dataset = driver.Open(outshp, 1)  # 1表示读写模式
        
        if shp_dataset is None:
            raise RuntimeError(f'Failed to open shapefile: {outshp}')
        
        layer = shp_dataset.GetLayer()

        features_to_delete = []
        feature = layer.GetNextFeature()
        while feature is not None:
            field_value = feature.GetField(field_name)
            if field_value == 0:
                features_to_delete.append(feature.GetFID())
            feature = layer.GetNextFeature()

        for fid in sorted(features_to_delete, reverse=True):
            layer.DeleteFeature(fid)
        
        layer.ResetReading()
        shp_dataset = None

    mem_raster = None
    band = None

# ==========================================================================================
# 随机划分样本，MHMAP工程版本，需要多个shp文件
# ==========================================================================================
def random_split_shp(input_shp: str, output_shp1: str, output_shp2: str, num_to_select: int | float, pixel_size: int = 29):
    """
    随机分割点Shapefile为两个新文件

    参数:
    input_shp: 输入点Shapefile路径
    output_shp1: 输出Shapefile1路径（包含随机选取的要素）
    output_shp2: 输出Shapefile2路径（包含剩余的要素）
    num_to_select: 要随机选取的要素数量, 如果小于1将按照比例选取
    """
    # 确保输入文件存在
    if not os.path.exists(input_shp):
        raise FileNotFoundError(f"输入文件不存在: {input_shp}")
    
    # 打开输入数据源
    driver = ogr.GetDriverByName('ESRI Shapefile')
    in_ds = driver.Open(input_shp, 0)
    if in_ds is None:
        raise RuntimeError(f"无法打开输入文件: {input_shp}")
    
    layer = in_ds.GetLayer()
    geom_type = layer.GetGeomType()
    geom_type_name = ogr.GeometryTypeToName(geom_type)
    # 判断是点矢量还是面矢量
    if geom_type_name == "Polygon":
        mask,geotransform,projection = Mianvector2mask(input_shp, fill_value=1, pixel_size=pixel_size)
        random.seed(42)
        ones_coords = np.argwhere(mask == 1)
        n_ones = len(ones_coords)
        n_first = int(n_ones * num_to_select) if num_to_select < 1 else num_to_select
        shuffled_indices = np.random.permutation(ones_coords)
        first_indices = shuffled_indices[:n_first]
        second_indices = shuffled_indices[n_first:]
        mask1 = np.zeros_like(mask)
        mask2 = np.zeros_like(mask)
        for x,y in first_indices:
            mask1[x,y] = 1
        for x,y in second_indices:
            mask2[x,y] = 1
        mask2poly(mask1, geotransform, projection, output_shp1)
        mask2poly(mask2, geotransform, projection, output_shp2)

        print(f"处理完成!")
        print(f"已随机选择 {n_first} 个像素区域保存到: {output_shp1}")
        print(f"剩余 {n_ones - n_first} 个像素区域保存到: {output_shp2}")
    else:
        # 获取要素总数
        total_features = layer.GetFeatureCount()
        if num_to_select > total_features:
            raise ValueError(f"要选择的要素数量({num_to_select})大于总要素数({total_features})")
        if num_to_select < 1:
            num_to_select = int(total_features * num_to_select) # 按比例选取
        # 生成随机索引列表
        indices = list(range(total_features))
        random.shuffle(indices)
        selected_indices = set(indices[:num_to_select])
        
        # 获取输入图层的空间参考和字段定义
        spatial_ref = layer.GetSpatialRef()
        layer_defn = layer.GetLayerDefn()
        
        # 创建输出数据源1（选中的要素）
        out_dir1 = os.path.dirname(output_shp1)
        if out_dir1 and not os.path.exists(out_dir1):
            os.makedirs(out_dir1, exist_ok=True)
        if os.path.exists(output_shp1):
            driver.DeleteDataSource(output_shp1)
        out_ds1 = driver.CreateDataSource(output_shp1)
        out_layer1 = out_ds1.CreateLayer(os.path.basename(output_shp1)[:-4], 
                                        spatial_ref, 
                                        ogr.wkbPoint)
        
        # 复制字段定义
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            out_layer1.CreateField(field_defn)
        
        # 创建输出数据源2（剩余的要素）
        out_dir2 = os.path.dirname(output_shp2)
        if out_dir2 and not os.path.exists(out_dir2):
            os.makedirs(out_dir2, exist_ok=True)
        if os.path.exists(output_shp2):
            driver.DeleteDataSource(output_shp2)
        out_ds2 = driver.CreateDataSource(output_shp2)
        out_layer2 = out_ds2.CreateLayer(os.path.basename(output_shp2)[:-4], 
                                        spatial_ref, 
                                        ogr.wkbPoint)
        
        # 复制字段定义
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            out_layer2.CreateField(field_defn)
        
        # 重置图层读取位置
        layer.ResetReading()
        
        # 遍历所有要素并根据索引分配到不同的输出文件
        for idx, in_feature in enumerate(layer):
            if idx in selected_indices:
                out_layer1.CreateFeature(in_feature.Clone())
            else:
                out_layer2.CreateFeature(in_feature.Clone())
        
        # 清理资源
        in_ds = None
        out_ds1 = None
        out_ds2 = None
        
        print(f"INFO: Train data {num_to_select} features saved to: {output_shp1}")
        print(f"INFO: Test data {total_features - num_to_select} features saved to: {output_shp2}")

def _split_records_n_way_by_weight(
    records: list[dict],
    ratios: list[float],
    seed: int = 42,
) -> list[list[dict]]:
    """
    按面积权重将要素记录分配到 N 个子集，分配比例由 ratios 指定。

    每个类别内部独立分配，确保各子集都尽量有该类别的样本。
    如果某类别要素数量少于分割数，多余子集可能为空。

    参数:
        records:  要素记录列表，每项含 'label'、'weight'、'feature' 键
        ratios:   各子集的比例，长度为 N，必须归一化（和为 1）
        seed:     随机种子

    返回:
        长度为 N 的列表，每项是属于该子集的 records 列表
    """
    n = len(ratios)
    rng = random.Random(seed)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec['label']].append(rec)

    subsets: list[list[dict]] = [[] for _ in range(n)]

    for _, cls_records in grouped.items():
        rng.shuffle(cls_records)
        total_weight = sum(r['weight'] for r in cls_records)
        if total_weight <= 0:
            total_weight = len(cls_records)

        # 计算每个子集应分配到的累计权重阈值（前缀和）
        thresholds = []
        cumulative = 0.0
        for ratio in ratios[:-1]:
            cumulative += ratio
            thresholds.append(total_weight * cumulative)

        current_weight = 0.0
        part_idx = 0
        for rec in cls_records:
            # 切换到下一分区
            while part_idx < n - 1 and current_weight >= thresholds[part_idx]:
                part_idx += 1
            subsets[part_idx].append(rec)
            current_weight += rec['weight']

        # 保证每个子集至少有一个来自该类别的要素（当要素数量足够时）
        n_cls = len(cls_records)
        for i in range(min(n, n_cls)):
            if not subsets[i]:
                # 从最大子集中借一个
                largest = max(range(n), key=lambda k: len(subsets[k]))
                if len(subsets[largest]) > 1:
                    subsets[i].append(subsets[largest].pop())

    return subsets


def batch_random_split_shp(
    input_shp_dir: str,
    output_dir: str,
    ratios: list[float],
    seed: int = 42,
    part_names: list[str] | None = None,
    valid_mask: np.ndarray | None = None,
    valid_mask_tif: str | gdal.Dataset | None = None,
) -> list[str]:
    """
    批量将 Shapefile 按给定比例列表随机分割为 N 个子集。

    - 点要素：按要素数量比例分配
    - 面要素：按多边形面积比例分配（不栅格化），与 random_split_shp_by_area_and_clip 一致

    输出结构（output_dir 下）:
        <part_names[0]>/  <part_names[1]>/  ... （默认为 split_part1, split_part2, ...）

    参数:
        input_shp_dir: 输入 Shapefile 文件夹路径，或单个 .shp 文件路径
        output_dir:    分割结果根目录
        ratios:        各子集比例列表，如 [0.6, 0.2, 0.2]，必须正数且和为 1
        seed:          随机种子，默认 42
        part_names:    各子集输出子文件夹名称列表，长度应与 ratios 一致；
                       未提供或长度不足时，多余分区使用 split_partN 作为默认名称

    返回:
        各子集输出目录的绝对路径列表
    """
    # ---------- 参数校验 ----------
    if not ratios or len(ratios) < 2:
        raise ValueError('ratios 至少需要 2 个元素')
    if any(r <= 0 for r in ratios):
        raise ValueError('ratios 中每个比例必须大于 0')
    total = sum(ratios)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f'ratios 之和必须为 1，当前为 {total:.6f}')

    n = len(ratios)
    os.makedirs(output_dir, exist_ok=True)

    valid_mask_ds = None
    close_valid_mask_ds = False
    if valid_mask is not None:
        valid_mask = valid_mask.astype(bool)
        if valid_mask_tif is None:
            raise ValueError('valid_mask_tif must be provided when valid_mask is used')
        if isinstance(valid_mask_tif, str):
            valid_mask_ds = gdal.Open(valid_mask_tif)
            close_valid_mask_ds = True
            if valid_mask_ds is None:
                raise RuntimeError(f'Cannot open valid_mask_tif: {valid_mask_tif}')
        elif isinstance(valid_mask_tif, gdal.Dataset):
            valid_mask_ds = valid_mask_tif
        else:
            raise TypeError('valid_mask_tif must be a path or gdal.Dataset')
        expected_shape = (valid_mask_ds.RasterYSize, valid_mask_ds.RasterXSize)
        if valid_mask.shape != expected_shape:
            raise ValueError(f'valid_mask shape {valid_mask.shape} does not match image shape {expected_shape}')

    def _weight_from_valid_mask(feature: ogr.Feature, source_layer: ogr.Layer, is_polygon_feature: bool) -> float:
        geom = feature.GetGeometryRef()
        if geom is None:
            return 0.0
        if valid_mask is None:
            if is_polygon_feature:
                area = geom.GetArea()
                return area if area > 0 else 1.0
            return 1.0

        geotrans = valid_mask_ds.GetGeoTransform()
        if not is_polygon_feature:
            geo_x, geo_y = geom.GetX(), geom.GetY()
            col = int((geo_x - geotrans[0]) / geotrans[1])
            row = int((geo_y - geotrans[3]) / geotrans[5])
            if 0 <= row < valid_mask.shape[0] and 0 <= col < valid_mask.shape[1] and valid_mask[row, col]:
                return 1.0
            return 0.0

        mem_driver = gdal.GetDriverByName('MEM')
        mask_ds = mem_driver.Create('', valid_mask.shape[1], valid_mask.shape[0], 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geotrans)
        mask_ds.SetProjection(valid_mask_ds.GetProjection())
        mask_band = mask_ds.GetRasterBand(1)
        mask_band.Fill(0)

        vec_driver = ogr.GetDriverByName('Memory')
        vec_ds = vec_driver.CreateDataSource('')
        vec_layer = vec_ds.CreateLayer('feature', srs=source_layer.GetSpatialRef(), geom_type=source_layer.GetGeomType())
        feat = ogr.Feature(vec_layer.GetLayerDefn())
        feat.SetGeometry(geom.Clone())
        vec_layer.CreateFeature(feat)
        feat = None

        gdal.RasterizeLayer(mask_ds, [1], vec_layer, burn_values=[1])
        feature_mask = mask_band.ReadAsArray().astype(bool)
        weight = int(np.sum(feature_mask & valid_mask))
        vec_ds = None
        mask_ds = None
        return float(weight)

    # 解析子文件夹名称：优先使用 part_names，不足时补 split_partN
    resolved_names: list[str] = []
    for i in range(n):
        if part_names and i < len(part_names) and part_names[i]:
            resolved_names.append(str(part_names[i]))
        else:
            resolved_names.append(f'split_part{i + 1}')

    # 创建所有子集目录
    part_dirs = [os.path.join(output_dir, name) for name in resolved_names]
    for d in part_dirs:
        os.makedirs(d, exist_ok=True)

    # 收集待处理的 shp 文件
    if os.path.isdir(input_shp_dir):
        shp_files = sorted(search_files_in_directory(input_shp_dir, extension='.shp'))
        if not shp_files:
            raise RuntimeError(f'目录中未找到 Shapefile: {input_shp_dir}')
    elif os.path.isfile(input_shp_dir):
        shp_files = [input_shp_dir]
    else:
        raise RuntimeError(f'无效路径（非文件夹也非 .shp 文件）: {input_shp_dir}')

    ratio_str = ' / '.join(f'{r:.0%}' for r in ratios)
    name_str  = ' / '.join(resolved_names)
    print(f'batch_random_split_shp | {len(shp_files)} 个文件 | [{name_str}] = [{ratio_str}] | seed={seed}')

    for shp_file in shp_files:
        base_name = os.path.basename(shp_file)[:-4]

        # 读取图层
        shp_ds = ogr.Open(shp_file)
        if shp_ds is None:
            print(f'  [!] 无法打开，跳过: {shp_file}')
            continue
        layer = shp_ds.GetLayer()
        geom_type = ogr.GT_Flatten(layer.GetGeomType())
        is_polygon = geom_type in (ogr.wkbPolygon, ogr.wkbMultiPolygon)

        # 构建 records（point → weight=1, polygon → weight=面积）
        records: list[dict] = []
        layer.ResetReading()
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            weight = _weight_from_valid_mask(feature, layer, is_polygon)
            if weight <= 0:
                continue
            # 如果有 class 字段则读取，否则统一 label=0
            class_field = _find_field_name_case_insensitive(
                layer, ['Classvalue', 'classvalue', 'class', 'label'])
            label_raw = feature.GetField(class_field) if class_field else None
            label = int(label_raw) if label_raw is not None else 0
            records.append({'feature': feature.Clone(), 'label': label, 'weight': weight})

        if not records:
            print(f'  [!] 无有效要素，跳过: {base_name}')
            shp_ds = None
            continue

        # 分割
        subsets = _split_records_n_way_by_weight(records, ratios, seed=seed)

        # 写出各子集
        counts = []
        for i, (part_records, part_dir) in enumerate(zip(subsets, part_dirs)):
            out_shp = os.path.join(part_dir, f'{base_name}_{resolved_names[i]}.shp')
            _write_feature_subset_to_shp(out_shp, layer, part_records)
            counts.append(len(part_records))

        count_str = ' | '.join(
            f'{resolved_names[i]}={c}{"(poly)" if is_polygon else "(pt)"}' for i, c in enumerate(counts))
        print(f'  {base_name}: {count_str}')
        shp_ds = None

    if close_valid_mask_ds:
        valid_mask_ds = None
    print(f'Complete. Output directory: {output_dir}')
    return part_dirs

# ==========================================================================================
# 根据GISPRO样本管理器创建的样本shp来随机划分训练-测试集并裁剪样本
# ==========================================================================================
def _find_field_name_case_insensitive(layer: ogr.Layer, candidates: list[str]) -> str | None:
    """从图层中按大小写不敏感方式查找字段名。"""
    layer_defn = layer.GetLayerDefn()
    field_name_map = {}
    for i in range(layer_defn.GetFieldCount()):
        name = layer_defn.GetFieldDefn(i).GetName()
        field_name_map[name.lower()] = name
    for candidate in candidates:
        match = field_name_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def _split_records_by_weight(records: list[dict], train_ratio: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    按类别分组后，使用要素面积或权重累计值进行训练/测试划分。
    这样可以在各类别内尽量维持与 train_ratio 一致的样本量比例。
    """
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec['label']].append(rec)

    train_records = []
    test_records = []

    for _, cls_records in grouped.items():
        rng.shuffle(cls_records)
        total_weight = sum(r['weight'] for r in cls_records)
        target_train_weight = total_weight * train_ratio
        current_train_weight = 0
        class_train = []
        class_test = []

        for rec in cls_records:
            rec_weight = rec['weight']
            if current_train_weight < target_train_weight:
                class_train.append(rec)
                current_train_weight += rec_weight
            else:
                class_test.append(rec)

        # 保证在类别内尽量两侧都有样本
        if len(class_test) == 0 and len(class_train) > 1:
            class_test.append(class_train.pop())
        if len(class_train) == 0 and len(class_test) > 1:
            class_train.append(class_test.pop())

        train_records.extend(class_train)
        test_records.extend(class_test)

    return train_records, test_records


def _resolve_dataset_split_ratios(
    train_ratio: float,
    val_ratio: float = 0.0,
    test_ratio: float | None = None
) -> tuple[float, float, float]:
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    else:
        test_ratio = float(test_ratio)

    ratios = (train_ratio, val_ratio, test_ratio)
    if train_ratio <= 0:
        raise ValueError('train_ratio 必须大于 0')
    if any(r < 0 for r in ratios):
        raise ValueError('train/val/test 比例不能为负数')
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(
            f'train/val/test 比例之和必须为 1，当前为 {sum(ratios):.6f}'
        )
    if val_ratio + test_ratio <= 0:
        raise ValueError('val_ratio 与 test_ratio 至少需要一个大于 0')
    return train_ratio, val_ratio, test_ratio


def _split_records_three_way_by_weight(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    train_records, holdout_records = _split_records_by_weight(records, train_ratio, seed=seed)

    if val_ratio <= 0:
        return train_records, [], holdout_records
    if test_ratio <= 0:
        return train_records, holdout_records, []

    holdout_ratio = val_ratio + test_ratio
    val_share = val_ratio / holdout_ratio
    val_records, test_records = _split_records_by_weight(holdout_records, val_share, seed=seed + 1)
    return train_records, val_records, test_records


def _write_feature_subset_to_shp(output_shp: str, source_layer: ogr.Layer, feature_records: list[dict]) -> str:
    """将要素子集写入新的 Shapefile，保留原字段结构。"""
    driver = ogr.GetDriverByName('ESRI Shapefile')
    out_dir = os.path.dirname(output_shp)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)

    source_layer_defn = source_layer.GetLayerDefn()
    source_srs = source_layer.GetSpatialRef()
    source_geom_type = source_layer.GetGeomType()

    old_shape_encoding = gdal.GetConfigOption('SHAPE_ENCODING')
    gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
    try:
        out_ds = driver.CreateDataSource(output_shp)
        if out_ds is None:
            raise RuntimeError(f'无法创建输出 Shapefile: {output_shp}')

        out_layer = out_ds.CreateLayer(
            os.path.basename(output_shp)[:-4],
            srs=source_srs,
            geom_type=source_geom_type,
            options=['ENCODING=UTF-8']
        )

        for i in range(source_layer_defn.GetFieldCount()):
            field_defn = source_layer_defn.GetFieldDefn(i)
            out_layer.CreateField(field_defn)

        for rec in feature_records:
            out_layer.CreateFeature(rec['feature'].Clone())

        out_ds.FlushCache()
        out_ds = None
    finally:
        gdal.SetConfigOption('SHAPE_ENCODING', old_shape_encoding)
    return output_shp


def _get_positions_from_feature(feature: ogr.Feature, tif_ds: gdal.Dataset,
                                rng: random.Random) -> list[list[int]]:
    """从单个点/面要素提取像素位置列表。"""
    geom = feature.GetGeometryRef()
    if geom is None:
        return []

    geotrans = tif_ds.GetGeoTransform()
    width = tif_ds.RasterXSize
    height = tif_ds.RasterYSize
    flat_type = ogr.GT_Flatten(geom.GetGeometryType())

    # 点要素: 一个点对应一个样本
    if flat_type == ogr.wkbPoint:
        geo_x, geo_y = geom.GetX(), geom.GetY()
        col = int((geo_x - geotrans[0]) / geotrans[1])
        row = int((geo_y - geotrans[3]) / geotrans[5])
        if 0 <= row < height and 0 <= col < width:
            return [[row, col]]
        return []

    # 面要素: 返回面内全部覆盖像素
    if flat_type in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
        mem_driver = gdal.GetDriverByName('MEM')
        mask_ds = mem_driver.Create('', width, height, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(geotrans)
        mask_ds.SetProjection(tif_ds.GetProjection())
        mask_band = mask_ds.GetRasterBand(1)
        mask_band.Fill(0)

        # Use RasterizeLayer for better compatibility across GDAL versions.
        vec_mem_driver = ogr.GetDriverByName('Memory')
        vec_ds = vec_mem_driver.CreateDataSource('')
        tif_srs = None
        tif_wkt = tif_ds.GetProjection()
        if tif_wkt:
            tif_srs = osr.SpatialReference()
            tif_srs.ImportFromWkt(tif_wkt)
        vec_layer = vec_ds.CreateLayer('poly', srs=tif_srs, geom_type=ogr.wkbUnknown)
        feat_defn = vec_layer.GetLayerDefn()
        feat = ogr.Feature(feat_defn)
        feat.SetGeometry(geom.Clone())
        vec_layer.CreateFeature(feat)
        feat = None

        gdal.RasterizeLayer(mask_ds, [1], vec_layer, burn_values=[1])
        mask = mask_band.ReadAsArray()
        coords = np.argwhere(mask == 1)
        if coords.size == 0:
            vec_ds = None
            mask_ds = None
            return []
        vec_ds = None
        mask_ds = None
        return coords.tolist()

    return []


def _clip_records_to_dataset_txt(records: list[dict], tif_ds: gdal.Dataset, out_dir: str,
                                 subset_name: str, patch_size: int, seed: int,
                                 output_format: str = 'tif') -> list[str]:
    """根据要素记录裁剪样本，并返回 'path label' 形式列表。"""
    rng = random.Random(seed)
    positions_by_label = defaultdict(list)

    for rec in records:
        positions = _get_positions_from_feature(rec['feature'], tif_ds, rng)
        if positions:
            positions_by_label[rec['label']].extend(positions)

    dataset_lines = []
    for label, positions in sorted(positions_by_label.items(), key=lambda x: x[0]):
        out_prefix = f'{subset_name}_label{label}'
        lines = clip_by_position(
            out_dir=out_dir,
            sr_img=tif_ds,
            position_list=positions,
            patch_size=patch_size,
            out_tif_name=out_prefix,
            label=label,
            output_format=output_format
        )
        dataset_lines.extend(lines)

    return dataset_lines


def random_split_shp_by_area_and_clip(
    tif_path: str,
    shp_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    test_ratio: float | None = None,
    patch_size: int = 9,
    seed: int = 42,
    output_format: str = 'tif'
) -> dict:
    """Split a single shapefile into train/val/test subsets and clip samples."""
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f'输入 tif 不存在: {tif_path}')
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f'输入 shp 不存在: {shp_path}')

    train_ratio, val_ratio, test_ratio = _resolve_dataset_split_ratios(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    output_format = str(output_format).lower()
    if output_format not in ('tif', 'bin'):
        raise ValueError(f"Unsupported output_format: {output_format}. Expected 'tif' or 'bin'.")

    os.makedirs(output_dir, exist_ok=True)
    split_dir = os.path.join(output_dir, 'split_shp')
    train_img_dir = os.path.join(output_dir, 'train_dataset')
    val_img_dir = os.path.join(output_dir, 'val_dataset')
    test_img_dir = os.path.join(output_dir, 'test_dataset')
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    shp_ds = ogr.Open(shp_path)
    if shp_ds is None:
        raise RuntimeError(f'无法打开输入 shp: {shp_path}')
    layer = shp_ds.GetLayer()

    class_field = _find_field_name_case_insensitive(layer, ['Classvalue', 'classvalue', 'class', 'label'])
    if class_field is None:
        shp_ds = None
        raise ValueError('未找到类别字段，请至少包含 Classvalue/class/label 之一')

    records = []
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        label_raw = feature.GetField(class_field)
        if label_raw is None:
            continue
        label = int(label_raw)

        flat_type = ogr.GT_Flatten(geom.GetGeometryType())
        if flat_type in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
            weight = geom.GetArea()
            if weight <= 0:
                weight = 1
        else:
            weight = 1

        records.append({
            'feature': feature.Clone(),
            'label': label,
            'weight': weight
        })

    if not records:
        shp_ds = None
        raise RuntimeError('输入 shp 中没有可用要素（请检查几何或字段值）')

    train_records, val_records, test_records = _split_records_three_way_by_weight(
        records=records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_shp = os.path.join(split_dir, 'train_split.shp')
    val_shp = os.path.join(split_dir, 'val_split.shp')
    test_shp = os.path.join(split_dir, 'test_split.shp')
    _write_feature_subset_to_shp(train_shp, layer, train_records)
    if val_ratio > 0:
        _write_feature_subset_to_shp(val_shp, layer, val_records)
    _write_feature_subset_to_shp(test_shp, layer, test_records)
    shp_ds = None

    tif_ds = gdal.Open(tif_path)
    if tif_ds is None:
        raise RuntimeError(f'无法打开输入 tif: {tif_path}')

    train_lines = _clip_records_to_dataset_txt(
        records=train_records,
        tif_ds=tif_ds,
        out_dir=train_img_dir,
        subset_name='train',
        patch_size=patch_size,
        seed=seed,
        output_format=output_format
    )
    val_lines = []
    if val_ratio > 0:
        val_lines = _clip_records_to_dataset_txt(
            records=val_records,
            tif_ds=tif_ds,
            out_dir=val_img_dir,
            subset_name='val',
            patch_size=patch_size,
            seed=seed + 1,
            output_format=output_format
        )
    test_lines = _clip_records_to_dataset_txt(
        records=test_records,
        tif_ds=tif_ds,
        out_dir=test_img_dir,
        subset_name='test',
        patch_size=patch_size,
        seed=seed + 2,
        output_format=output_format
    )

    train_txt = os.path.join(train_img_dir, '.datasets.txt')
    val_txt = os.path.join(val_img_dir, '.datasets.txt')
    test_txt = os.path.join(test_img_dir, '.datasets.txt')
    write_list_to_txt(train_lines, train_txt)
    write_list_to_txt(val_lines, val_txt)
    write_list_to_txt(test_lines, test_txt)

    tif_ds = None

    print(f'train shapefile: {train_shp}')
    if val_ratio > 0:
        print(f'val shapefile: {val_shp}')
    print(f'test shapefile: {test_shp}')
    print(f'train samples txt: {train_txt} ({len(train_lines)} samples)')
    if val_ratio > 0:
        print(f'val samples txt: {val_txt} ({len(val_lines)} samples)')
    print(f'test samples txt: {test_txt} ({len(test_lines)} samples)')

    return {
        'train_shp': train_shp,
        'val_shp': val_shp if val_ratio > 0 else None,
        'test_shp': test_shp,
        'train_txt': train_txt,
        'val_txt': val_txt,
        'test_txt': test_txt,
        'train_samples': len(train_lines),
        'val_samples': len(val_lines),
        'test_samples': len(test_lines)
    }
# ==========================================================================================
# 分类图去除碎斑（栅格格式）
# ==========================================================================================
def sieve_filtering(input_tif_path, output_tif_path, threshold_pixels, connectedness=8, mask=None):
    """
    使用GDAL的SieveFilter去除碎斑
    :param input_tif_path: 输入的分类结果TIF路径
    :param output_tif_path: 输出的平滑后TIF路径
    :param threshold_pixels: 阈值，小于该像素数的连通域将被合并 (例如 50)
    :param connectedness: 连通性，4代表上下左右，8代表包含对角线 (通常用8)
    :param mask: 掩膜，指定处理区域，None表示处理所有像素
    """
    if isinstance(input_tif_path, gdal.Dataset):
        src_ds = input_tif_path
    else:
        src_ds = gdal.Open(input_tif_path, gdal.GA_ReadOnly)
        if src_ds is None:
            raise FileNotFoundError(f"无法打开文件: {input_tif_path}")
        
    src_band = src_ds.GetRasterBand(1)

    driver = gdal.GetDriverByName('GTiff')
    # 创建副本作为输出，避免修改原图
    out_dir = os.path.dirname(output_tif_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    dst_ds = driver.CreateCopy(output_tif_path, src_ds, 0)
    dst_band = dst_ds.GetRasterBand(1)
    if mask is not None:
        if mask.dtype == np.bool_:
            mask = mask.astype(np.uint8)
        mem_driver = gdal.GetDriverByName('MEM')
        tmp_mask_ds = mem_driver.Create('', mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte)
        tmp_mask_band = tmp_mask_ds.GetRasterBand(1)
        tmp_mask_band.WriteArray(mask)
        mask_to_use = tmp_mask_band

    print(f"正在处理碎斑... 阈值: {threshold_pixels} 像素")
    gdal.SieveFilter(
        src_band,       # 输入波段
        mask_to_use,           # 掩膜 (None表示处理所有像素)
        dst_band,       # 输出波段 (直接写入到输出文件)
        threshold_pixels, 
        connectedness, 
        [],             # 额外参数
        None,           # 进度回调函数
        None            # 回调数据
    )

    dst_ds.FlushCache() # 确保数据写入磁盘
    src_ds = None
    dst_ds = None
    print(f"处理完成，结果已保存至: {output_tif_path}")

if __name__ == '__main__':
    pass
