import os.path
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
from utils import search_files_in_directory, write_list_to_txt

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

def write_data_to_tif(output_file, data, geotransform, projection, nodata_value=None, mask=None):
    """
    将数组数据写入GeoTIFF文件
    
    参数:
        output_file (str): 输出文件路径
        data (np.ndarray): 三维数组（波段,行,列）
        geotransform (tuple): 6参数地理变换
        projection (str): WKT格式的坐标参考系统
        nodata_value (int/float): NoData值, 默认0
        mask (np.ndarray): 可选的掩码数组, 形状应与data的空间维度匹配, 用于保存有效数据区域，其余区域设为nodata_value
    
    异常:
        IOError: 当文件创建失败时
    """
    # 处理二维数组情况（单波段）
    if len(data.shape) == 2:
        rows, cols = data.shape
        bands = 1
        data = data.reshape((1, rows, cols))  # 转换为三维
    elif len(data.shape) == 3:
        bands, rows, cols = data.shape
    else:
        raise ValueError("输入数据必须是二维或三维数组")
    
    try:
        dtype = NP2GDAL_TYPE[data.dtype]
    except KeyError:
        raise ValueError(f"不支持的数据类型: {data.dtype}. 支持的类型包括: {list(NP2GDAL_TYPE.keys())}")
    if mask is not None: # 如果提供了掩码, 则将掩码区域外的数据设为nodata_value（默认为0）
        mask = mask.astype(np.bool)
        data = data.transpose((1, 2, 0))
        data[~mask] = nodata_value
        data = data.transpose((2, 0, 1))
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_file, cols, rows, bands, dtype)
    if dataset is None:
        raise IOError(f"无法创建文件 {output_file}")
    # 设置地理变换和投影
    if geotransform is not None and projection is not None:
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(projection)
    # 写入数据
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(data[i,:,:])
        if nodata_value is not None:
            band.SetNoDataValue(nodata_value)  # 设置 NoData 值
    # 释放资源
    dataset.FlushCache()
    dataset = None
    return output_file

def read_tif(tif_path):
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
    

def Mianvector2mask(vector_path, tif_path=None, fill_value=255, pixel_size=1.0):
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

def mask_to_point_shp(mask_matrix, tif_path, output_shapefile="./Position_mask/test.shp"):
    """
    将二维矩阵mask矩阵转化为矢量点文件
    """
    # 获取矩阵的行和列
    geotransform, projection, _, _, _ = read_tif(tif_path)
    rows, cols = mask_matrix.shape
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if not driver:
        raise RuntimeError("Shapefile driver not available")
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

def mask_to_multipoint_shp(mask_matrix, tif_path, output_dir="./Position_mask", output_dir_name=None):
    """
    在指定文件夹下创建一个新文件夹保存样本的矢量点文件
    """
    labels = np.unique(mask_matrix)
    if output_dir_name is None:
        output_dir_name = 'SAMPLES_DIR'
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

def point_shp_to_mask(shapefile, tif_path, value=None):
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

def mutipoint_shp_to_mask(shapefile_dir, tif_path):
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

def clip_by_position(out_dir, sr_img, position_list, patch_size=30, out_tif_name='img', label=None):
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
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) * 1e-4 # 如何data是int类型, 进行放缩并转化为float32类型
                    offset_x = read_x - x_start
                    offset_y = read_y - y_start
                    full_data[offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
        
            # 计算新的地理变换
            new_geotrans = list(im_geotrans)
            new_geotrans[0] = im_geotrans[0] + x_start * im_geotrans[1]
            new_geotrans[3] = im_geotrans[3] + y_start * im_geotrans[5]
            
            # 保存结果
            idx += 1
            out_path = os.path.join(out_dir, f"{out_tif_name}_{idx}.tif")
            write_data_to_tif(out_path, full_data, new_geotrans, im_proj)
            if label is not None:
                out_dataset.append(f"{out_path} {label}")
        pbar.update(1)
    pbar.close()
    if need_close:
        im_dataset = None
    return out_dataset

def clip_by_shp(out_dir, sr_img, shp_path, patch_size=30, out_tif_name='img', label=None, sample_num=10000):
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
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize

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
    out_dataset = clip_by_position(out_dir, sr_img, positions, patch_size, out_tif_name=out_tif_name, label=label)
    im_dataset = None
    return out_dataset

def clip_by_multishp(out_dir, sr_img, shp_dir, block_size=30, out_tif_name='img'):
    """
    批量处理目录下多个Shapefile的裁剪任务, 并自动生成记录样本块与标签的数据集
    
    参数:
        out_dir (str): 输出目录路径
        sr_img (str/gdal.Dataset): 输入影像路径或已打开的GDAL数据集对象  
        shp_dir (str): 包含点Shapefiles的目录路径或者是一个Shapefile文件路径
        block_size (int): 裁剪块大小（像素）, 默认30
        out_tif_name (str): 输出文件名前缀, 默认'img'
        fill_value (int/float): 边缘填充值, 默认0
    
    返回:
        None
    
    异常:
        RuntimeError: 当目录中没有Shapefile或裁剪失败时
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.isdir(shp_dir):
        point_shp_files = search_files_in_directory(shp_dir, extension='.shp')
        if not point_shp_files:
            raise RuntimeError(f'Not shapefiles found in directory: {shp_dir}')
        all_out_datasets = []
        for idx, point_shp in enumerate(point_shp_files):
            out_dataset = clip_by_shp(out_dir, sr_img, point_shp, block_size, out_tif_name=f'{out_tif_name}_label{idx}', label=idx)
            all_out_datasets.extend(out_dataset)
        if not all_out_datasets:
            pass
        else:
            dataset_path = os.path.join(out_dir, '.datasets.txt')
            write_list_to_txt(all_out_datasets, dataset_path)
            print(f'dataset file saved to: {dataset_path}')
    elif os.path.isfile(shp_dir):
        print('single shape file dont need to write dataset file')
        out_dataset = clip_by_shp(out_dir, sr_img, shp_dir, block_size, out_tif_name=out_tif_name)
    else:
        raise RuntimeError(f'Invalid point_shp_dir: {shp_dir}, it should be a directory or a shapefile path')
    

def batch_raster_to_vector(tif_dir, shp_img_path, extension='.tif', dict=None, delete_value=0, if_smooth=False):
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

def mask2poly(mask_array, geotransform, projection, outshp, field_name='value', remove_zero=True):
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
    # 创建内存中的栅格数据集
    driver = gdal.GetDriverByName('MEM')
    rows, cols = mask_array.shape
    
    # 创建内存栅格
    mem_raster = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    mem_raster.SetGeoTransform(geotransform)
    mem_raster.SetProjection(projection)
    
    # 写入数据
    band = mem_raster.GetRasterBand(1)
    band.WriteArray(mask_array)
    band.SetNoDataValue(0)
    band.FlushCache()
    
    # 创建空间参考
    prj = osr.SpatialReference()
    prj.ImportFromWkt(projection)
    
    # 创建Shapefile
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):
        drv.DeleteDataSource(outshp)
    
    polygon_ds = drv.CreateDataSource(outshp)
    poly_layer = polygon_ds.CreateLayer(
        os.path.basename(outshp)[:-4],  # 图层名（去掉.shp后缀）
        srs=prj, 
        geom_type=ogr.wkbMultiPolygon
    )
    
    # 创建属性字段
    field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
    poly_layer.CreateField(field_defn)
    
    # 执行栅格转矢量
    gdal.Polygonize(band, None, poly_layer, 0, [], callback=None)
    
    # 同步到磁盘
    poly_layer.SyncToDisk()
    polygon_ds = None
    
    # 如果需要删除值为0的要素
    if remove_zero:
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shp_dataset = driver.Open(outshp, 1)  # 1表示读写模式
        
        if shp_dataset is None:
            raise RuntimeError(f'Failed to open shapefile: {outshp}')
        
        layer = shp_dataset.GetLayer()
        
        # 删除值为0的要素
        features_to_delete = []
        feature = layer.GetNextFeature()
        while feature is not None:
            field_value = feature.GetField(field_name)
            if field_value == 0:
                features_to_delete.append(feature.GetFID())
            feature = layer.GetNextFeature()
        
        # 删除要素（从后往前删除，避免索引变化）
        for fid in sorted(features_to_delete, reverse=True):
            layer.DeleteFeature(fid)
        
        layer.ResetReading()
        shp_dataset = None
    
    # 清理内存栅格
    mem_raster = None
    band = None

def random_split_shp(input_shp, output_shp1, output_shp2, num_to_select, pixel_size=29):
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
        
        print(f"已随机选择 {num_to_select} 个要素保存到: {output_shp1}")
        print(f"剩余 {total_features - num_to_select} 个要素保存到: {output_shp2}")

def batch_random_split_shp(input_shp_dir, output_dir, num_to_select, pixel_size=29):
    """
    批量随机分割点Shapefile为两个新文件
    
    参数:
    input_shp_dir: 输入点Shapefile目录路径
    output_dir: 输出目录路径，包含分割后的Shapefiles
    num_to_select: 要随机选取的要素数量
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.isdir(input_shp_dir):
        shp_files = search_files_in_directory(input_shp_dir, extension='.shp')
        output_dir1 = os.path.join(output_dir, "split_part1")
        output_dir2 = os.path.join(output_dir, "split_part2")
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
            os.makedirs(output_dir2)
        if not shp_files:
            raise RuntimeError(f'Not shapefiles found in directory: {input_shp_dir}')
        for shp_file in shp_files:
            base_name = os.path.basename(shp_file)[:-4]
            output_shp1 = os.path.join(output_dir1, f"{base_name}_part1.shp")
            output_shp2 = os.path.join(output_dir2, f"{base_name}_part2.shp")
            random_split_shp(shp_file, output_shp1, output_shp2, num_to_select, pixel_size=pixel_size)
    elif os.path.isfile(input_shp_dir):
        base_name = os.path.basename(input_shp_dir)[:-4]
        output_shp1 = os.path.join(output_dir, f"{base_name}_part1.shp")
        output_shp2 = os.path.join(output_dir, f"{base_name}_part2.shp")
        random_split_shp(input_shp_dir, output_shp1, output_shp2, num_to_select, pixel_size=pixel_size)
    else:
        raise RuntimeError(f'Invalid input_shp_dir: {input_shp_dir}, it should be a directory or a shapefile path')

if __name__ == '__main__':
    clip_by_multishp(
        out_dir=r'c:\Users\85002\Desktop\构造解译示意\test2',
        sr_img=r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1_3bands.dat',
        shp_dir=r'c:\Users\85002\OneDrive\文档\小论文\dataset11classes\d6-4new\split_part1',
        block_size=13,
        out_tif_name='img'
    )