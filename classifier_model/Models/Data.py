import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
import json
import random
try:
    import h5pickle
    import lmdb
except:
    pass

try:
    from osgeo import gdal, ogr
    gdal.UseExceptions()
except ImportError as e:
    gdal = None
    ogr = None
    print(f"GDAL 导入失败，真正的错误原因是: {e}")
    
_BIN_META_CACHE = {}

def _load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _find_bin_meta(bin_path):
    # According to gdal_utils clip output, metadata is always in the same
    # folder as .bin samples with fixed name raw_bip_meta.json.
    abs_path = os.path.abspath(bin_path)
    bin_dir = os.path.dirname(abs_path)
    if bin_dir in _BIN_META_CACHE:
        return _BIN_META_CACHE[bin_dir]

    meta_path = os.path.join(bin_dir, "raw_bip_meta.json")
    obj = _load_json_if_exists(meta_path)
    meta = obj if isinstance(obj, dict) else None
    if meta is not None and not all(k in meta for k in ("bands", "height", "width", "dtype")):
        meta = None

    _BIN_META_CACHE[bin_dir] = meta
    return meta

def _read_raw_bin(bin_path, meta):
    if meta is None:
        raise ValueError(
            f"Cannot auto-read raw bin without metadata: {bin_path}. "
            "Please provide raw_bip_meta.json in the same bin directory."
        )
    bands = int(meta["bands"])
    height = int(meta["height"])
    width = int(meta["width"])
    dtype = np.dtype(str(meta["dtype"]))
    layout = str(meta.get("layout", "BHW")).upper()
    numel = bands * height * width
    flat = np.fromfile(bin_path, dtype=dtype, count=numel)
    if flat.size != numel:
        raise ValueError(
            f"Invalid bin size for {bin_path}: got {flat.size}, expected {numel}"
        )
    if layout in ("BHW", "CHW"):
        arr = flat.reshape((bands, height, width))
    elif layout in ("HWB", "HWC"):
        arr = flat.reshape((height, width, bands))
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported raw bin layout: {layout}")
    return np.ascontiguousarray(arr)

def read_hsi_auto(image_path):
    """Auto-detect sample format and return [bands, H, W]."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".bin":
        return _read_raw_bin(image_path, _find_bin_meta(image_path))

    if gdal is not None:
        ds = gdal.Open(image_path)
        if ds is not None:
            arr = ds.ReadAsArray()
            ds = None
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            return np.ascontiguousarray(arr)
    raise RuntimeError(f"Unsupported or unreadable sample format: {image_path}")


def read_data(tif_path):
    """兼容旧接口：现在自动识别格式。"""
    return read_hsi_auto(tif_path)

class CNN_Dataset(Dataset):
    '''用于3D——CNN训练和测试'''
    def __init__(self, data_list, transform=None):
        """
        将列表划分为数据集
        Args:
            transform (callable, optional): 图像变换操作
        """
        self.data_list = data_list
        self.transform = transform
        image, label = self._get_example_(0)
        self.data_shape = image.shape  # 获取数据形状
        self.scale = 1e-4 if (image.abs().max() > 100) else 1.0

    def __len__(self):
        return len(self.data_list)
    
    def _get_example_(self, idx): # 获取单个样本的图像和标签
        image_path, label = self.data_list[idx].split()
        label = np.array(label, dtype=np.int16)
        image = read_data(image_path)
        bands, h, w = image.shape
        if h == 1 and w == 1:  # 自动识别1D 输入
            image = image.squeeze() # (bands,)
        label = torch.tensor(label, dtype=torch.long)
        image = torch.tensor(image, dtype=torch.float32)
        return image, label
    
    def __getitem__(self, idx):
        """
        根据索引返回图像及其标签
        image（1，C，H，W）
        """
        image_path, label = self.data_list[idx].split()
        label = np.array(label, dtype=np.int16)
        image = read_data(image_path)
        bands, h, w = image.shape
        if h == 1 and w == 1:  # 自动识别1D 输入
            image = image.squeeze() # (bands,)
        label = torch.tensor(label, dtype=torch.long)
        image = torch.tensor(image, dtype=torch.float32) * self.scale
        if self.transform:
            image = self.transform(image)
        return image, label

class Predict_Dataset(Dataset):
    '''构造用于3D编码器的输入''' 
    def __init__(self, patch_size, transform=None):
        self.block_size = patch_size
        self.data = None
        self.rows, self.cols = 0, 0
        self.idx = None
        self.scale = None
        self.transform = transform

    def __len__(self):
        return len(self.idx) if self.idx is not None else 0
    
    def __getitem__(self, idx):
        """
        根据索引返回图像及其光谱
        """
        index = self.idx[idx]
        row = index // self.cols
        col = index % self.cols # 根据索引生成二维索引
        block = self.get_samples(row, col)
        # 转换为 PyTorch 张量
        scale = self.scale if self.scale is not None else 1.0
        block = torch.tensor(block, dtype=torch.float32) * scale
        if self.transform:
            block = self.transform(block)
        return block

    def get_samples(self,row,col):
        block = self.data[:,row:row + self.block_size, col:col + self.block_size]
        if self.block_size == 1: # 如果是单像素，数据适配1D CNN的输入
            block = block.squeeze()
        return block
    
    def update_data(self, data, background_mask=None):
        """每个 block 更新数据"""
        rows, cols = background_mask.shape
        self.data = data
        self.rows = rows
        self.cols = cols
        self.idx = np.arange(self.rows * self.cols)

        mask = background_mask.astype(np.bool_)
        self.idx = self.idx.reshape(self.rows, self.cols)
        self.idx = self.idx[mask]
        self.scale = 1e-4 if (np.abs(self.data).max() > 100) else 1.0


class Predict_Shared_Dataset(Dataset):
    """
    在原 Predict_Dataset 基础上提供共享内存版本：
    - 仍然使用 update_data(block, mask) 更新当前块
    - 可配合 persistent_workers=True，避免每个 block 重启 worker
    """

    def __init__(self, patch_size, bands, max_block_size, transform=None):
        self.block_size = int(patch_size)
        self.bands = int(bands)
        self.max_block_size = int(max_block_size)
        self.transform = transform

        if self.block_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.max_block_size < 1:
            raise ValueError("max_block_size must be >= 1")
        if self.bands < 1:
            raise ValueError("bands must be >= 1")

        # image_block_iter 返回的数据尺寸上限：block_size + patch_size - 1
        self.max_data_h = self.max_block_size + self.block_size - 1
        self.max_data_w = self.max_block_size + self.block_size - 1
        self.max_coords = self.max_block_size * self.max_block_size

        # 共享内存：worker 持久化后可看到主进程 update_data 的更新
        self.shared_data = torch.zeros(
            (self.bands, self.max_data_h, self.max_data_w), dtype=torch.float32
        ).share_memory_()
        self.shared_coords = torch.zeros((self.max_coords, 2), dtype=torch.int32).share_memory_()
        self.shared_count = torch.zeros(1, dtype=torch.int32).share_memory_()

        self.scale = 1.0

    def __len__(self):
        return int(self.shared_count[0].item())

    def __getitem__(self, idx):
        row = int(self.shared_coords[idx, 0].item())
        col = int(self.shared_coords[idx, 1].item())
        block = self.shared_data[:, row:row + self.block_size, col:col + self.block_size]
        if self.block_size == 1:
            block = block.squeeze(-1).squeeze(-1)
        if self.transform:
            block = self.transform(block)
        return block

    def update_data(self, data, background_mask=None):
        """
        更新当前 block 数据到共享内存。
        data: numpy [bands, H, W]
        background_mask: numpy [rows, cols], bool
        """
        if background_mask is None:
            raise ValueError("background_mask cannot be None")
        if data.ndim != 3:
            raise ValueError(f"Expect data shape [bands,H,W], got {data.shape}")

        rows, cols = background_mask.shape
        _, h, w = data.shape
        if h > self.max_data_h or w > self.max_data_w:
            raise ValueError(
                f"block data too large: {(h, w)}, max allowed {(self.max_data_h, self.max_data_w)}"
            )
        if rows > self.max_block_size or cols > self.max_block_size:
            raise ValueError(
                f"mask too large: {(rows, cols)}, max allowed {(self.max_block_size, self.max_block_size)}"
            )

        scale = 1e-4 if (np.abs(data).max() > 100) else 1.0
        self.scale = scale
        # 一次性转换为 float 并缩放，__getitem__ 内不再重复转换
        tensor_data = torch.from_numpy(np.ascontiguousarray(data)).float() * scale
        self.shared_data.zero_()
        self.shared_data[:, :h, :w].copy_(tensor_data)

        coords = np.argwhere(background_mask.astype(bool))
        count = int(coords.shape[0])
        if count > self.max_coords:
            raise ValueError(f"too many coords: {count}, max allowed {self.max_coords}")
        self.shared_coords.zero_()
        if count > 0:
            self.shared_coords[:count].copy_(torch.from_numpy(coords.astype(np.int32)))
        self.shared_count[0] = count
    
class MoniHDF5_leaning_dataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.hf = h5pickle.File(self.h5_file, 'r',rdcc_nbytes=512 * 1024**2)  # 只读模式
        self.data = self.hf['data']
        self.labels = self.hf['labels']

    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)

    def __getitem__(self, index):
        """
        获取单个样本
        :param index: 样本索引
        :return: (影像数据, 标签)
        """
        img = self.data[index]  # 读取 HDF5 数据
        label = self.labels[index]  # 读取标签
        img = torch.tensor(img, dtype=torch.float32)  # 转换为 Tensor
        label = torch.tensor(label, dtype=torch.long)  # 转换为 Tensor
        return img, label

    def close(self):
        """关闭 HDF5 文件"""
        self.hf.close()


# ==============================================================================
# 对比学习
# ==============================================================================
class Contrastive_Dataset(Dataset):
    '''输入一个list文件, list元素代表数据地址'''
    def __init__(self, data_list,  
                 patch_size = None,
                 multith_mode = False):
        """
        将列表划分为数据集,[batch, 1, H, w, bands]
        """
        self.image_paths = data_list
        image = self._get_example_(0)
        self.data_shape = image.shape
        self.scale = 1e-4 if (image.abs().max() > 100) else 1.0

    def __len__(self):
        return len(self.image_paths)
    
    def _get_example_(self, idx):
        image_path = self.image_paths[idx]
        image = read_data(image_path)
        bands, h, w = image.shape
        if h == 1 and w == 1:  # 自动识别1D 输入
            image = image.squeeze() # (bands,)
        image = torch.tensor(image, dtype=torch.float32)
        return image
    
    def __getitem__(self, idx):
        """
        根据索引返回图像及其标签
        image(3, rows, cols)
        """
        image_path = self.image_paths[idx]
        image = read_data(image_path)
        bands, h, w = image.shape
        if h == 1 and w == 1:  # 自动识别1D 输入
            image = image.squeeze() # (bands,)
        image = torch.tensor(image, dtype=torch.float32) * self.scale
        return image
    
    def reset(self):
        pass

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        """
        Args:
            lmdb_path (str): LMDB 数据库文件夹路径
            transform (callable, optional): 可选的数据预处理/增强函数
        """
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None
        self.txn = None
        
        # 预先读取数据集长度与数据形状
        temp_env = lmdb.open(
            lmdb_path, 
            max_readers=1, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False
        )
        
        try:
            with temp_env.begin(write=False) as txn:
                # 获取长度
                length_bytes = txn.get(b'__len__')
                if length_bytes is None:
                    raise ValueError("未在 LMDB 中找到 '__len__' 键。")
                self.length = int(length_bytes.decode('ascii'))
                # 获取第一个样本以计算 Shape 和 Scale
                first_key = f"img_0".encode('ascii')
                byte_data = txn.get(first_key)
                if byte_data is None:
                    raise IndexError("无法读取 img_0 用于初始化，请检查数据集是否为空。")
                
                img_numpy = pickle.loads(byte_data)
                img_numpy = np.copy(img_numpy) # Copy to memory
                sample_tensor = torch.from_numpy(img_numpy).float()
                self.data_shape = sample_tensor.shape
                self.scale = 1e-4 if (sample_tensor.abs().max() > 100) else 1.0
                print(f"Dataset Loaded. Length: {self.length}, Shape: {self.data_shape}, Scale factor: {self.scale}")
                
        finally:
            temp_env.close() # 关闭临时连接

    def _init_env(self):
        """
        初始化 LMDB 环境。
        这个函数会在 __getitem__ 中被调用，确保每个 Worker 进程有独立的连接。
        """
        self.env = lmdb.open(
            self.lmdb_path, 
            max_readers=1, 
            readonly=True, 
            lock=False, 
            readahead=False, 
            meminit=False
        )
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 懒加载：如果当前进程还没有打开环境，则打开
        if self.env is None:
            self._init_env()
        key = f"img_{index}".encode('ascii') # key 从0开始
        byte_data = self.txn.get(key)

        if byte_data is None:
            # 防御性编程：理论上不应发生，除非 index 越界或数据库损坏
            raise IndexError(f"Key {key} not found in LMDB.")
        # 反序列化
        img_numpy = pickle.loads(byte_data)
        # LMDB 返回的是只读 buffer，PyTorch 无法直接从只读内存创建 Tensor (会有负步长或不可写问题)
        # 必须 copy 一份到内存
        img_numpy = np.copy(img_numpy)
        if img_numpy.ndim == 2:
            img_numpy = img_numpy[np.newaxis, :, :] # (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img_numpy).float() * self.scale
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor
    
    def reset(self):
        pass
    
# class Dataset_3D_H5(Dataset):
#     def __init__(self, h5_file):
#         self.h5_file = h5_file
#         self.hf = h5pickle.File(self.h5_file, 'r')  # 只读模式
#         self.data = self.hf['data']
#     def __len__(self):
#         """返回数据集大小"""
#         return len(self.data)
#     def __getitem__(self, index):
#         """
#         获取单个样本
#         :param index: 样本索引
#         :return: (影像数据, 标签)
#         """
#         img = self.data[index]  # 读取 HDF5 数据
#         # img = torch.tensor(img, dtype=torch.float32)  # 转换为 Tensor
#         return img
#     def close(self):
#         """关闭 HDF5 文件"""
#         self.hf.close()


class DynamicCropDataset(Dataset):
    """
    动态从原始影像和点Shapefile生成训练数据的Dataset
    
    参数:
        image_path (str): 原始影像路径
        point_shp (str): 点Shapefile路径
        block_size (int): 裁剪块大小(像素)
        transform (callable): 可选的数据增强变换
        fill_value (int/float): 边缘填充值
    """
    
    def __init__(self, image_path, point_shp, patch_size=30, 
                 transform=None, fill_value=0):
        self.image_path = image_path
        self.point_shp = point_shp
        self.patch_size = patch_size
        self.transform = transform
        self.fill_value = fill_value
        
        # 初始化GDAL资源
        self.im_dataset = gdal.Open(image_path)
        if self.im_dataset is None:
            raise RuntimeError(f"无法打开影像文件: {image_path}")
            
        # 获取影像基本信息
        self.im_geotrans = self.im_dataset.GetGeoTransform()
        self.im_proj = self.im_dataset.GetProjection()
        self.im_width = self.im_dataset.RasterXSize
        self.im_height = self.im_dataset.RasterYSize
        self.im_bands = self.im_dataset.RasterCount
        
        # 加载所有有效点坐标
        self.point_coords = self._load_point_coordinates()

        # 获取数据形状
        image = self.__getitem__(0)
        self.data_shape = image.shape 

    
    def _load_point_coordinates(self):
        """依次加载所有有效的点坐标(影像范围内的点),为shp文件创建索引字段Emb_Idx,字段0为无编码点,字段1-n为有编码点"""
        coords = []
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shp_dataset = driver.Open(self.point_shp, 1)  # 1 表示可写
        if shp_dataset is None:
            raise RuntimeError(f"无法打开矢量文件: {self.point_shp}")
            
        layer = shp_dataset.GetLayer()
        # 检查并删除现有Emb_Idx字段
        field_idx = layer.FindFieldIndex('Emb_Idx', 1)
        if field_idx != -1:
            layer.DeleteField(field_idx)
            print("已删除现有Emb_Idx字段")
        
        # 创建新的Emb_Idx字段
        embedding_field = ogr.FieldDefn('Emb_Idx', ogr.OFTInteger)
        if layer.CreateField(embedding_field) != 0:
            raise RuntimeError("创建Emb_Idx字段失败")
        # 获取新字段的索引
        layer_defn = layer.GetLayerDefn()
        field_idx = layer_defn.GetFieldIndex('Emb_Idx')
        if field_idx == -1:
            raise RuntimeError("无法找到新创建的 Emb_Idx 字段")
        idx = 1
        layer.ResetReading()  # 重置读取位置
        for feature in layer:
            geom = feature.GetGeometryRef()
            geoX, geoY = geom.GetX(), geom.GetY()
            
            # 转换为像素坐标
            x = int((geoX - self.im_geotrans[0]) / self.im_geotrans[1])
            y = int((geoY - self.im_geotrans[3]) / self.im_geotrans[5])
            
            # 检查是否在影像范围内
            if (0 <= x < self.im_width and 
                0 <= y < self.im_height):
                coords.append((x, y))
                feature.SetField(field_idx, idx) # 设置点的编码索引
                idx += 1
            else: 
                feature.SetField(field_idx, 0)
            layer.SetFeature(feature)
        print('Has set Emb_Idx field')
        shp_dataset = None
        if not coords:
            raise RuntimeError("没有找到影像范围内的有效点")
        return coords
    
    def __len__(self):
        return len(self.point_coords)
    
    def __getitem__(self, idx):
        """动态裁剪并返回数据块"""
        x, y = self.point_coords[idx]
        
        # 计算裁剪窗口
        if self.patch_size % 2 == 0:
            left_top = self.patch_size // 2 - 1
            right_bottom = self.patch_size // 2
        else:
            left_top = right_bottom = self.patch_size // 2
            
        x_start = x - left_top
        y_start = y - left_top
        x_end = x + right_bottom + 1
        y_end = y + right_bottom + 1
        
        # 计算实际可读取范围
        read_x = max(0, x_start)
        read_y = max(0, y_start)
        read_width = min(x_end, self.im_width) - read_x
        read_height = min(y_end, self.im_height) - read_y
        
        # 创建填充数组
        if self.im_bands > 1:
            block = np.full((self.im_bands, self.patch_size, self.patch_size), 
                           self.fill_value, dtype=np.float32)
        else:
            block = np.full((self.patch_size, self.patch_size), 
                          self.fill_value, dtype=np.float32)
        
        # 读取并填充有效数据
        if read_width > 0 and read_height > 0:
            if self.im_bands > 1:
                data = self.im_dataset.ReadAsArray(read_x, read_y, read_width, read_height)
                offset_x = read_x - x_start
                offset_y = read_y - y_start
                block[:, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
            else:
                data = self.im_dataset.GetRasterBand(1).ReadAsArray(read_x, read_y, read_width, read_height)
                offset_x = read_x - x_start
                offset_y = read_y - y_start
                block[offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
        
        # 转换为torch张量
        block = torch.tensor(block, dtype=torch.float32)
        if self.patch_size == 1:
            block = block.squeeze()
        
        # 应用数据增强
        if self.transform:
            block = self.transform(block)
            
        return block
    
    def __del__(self):
        """释放GDAL资源"""
        if hasattr(self, 'im_dataset') and self.im_dataset:
            self.im_dataset = None



class Im_info(object):
    """用于计算影像掩膜, 记录影像信息"""
    def __init__(self, image_path, multith_mode=False):
        self.image_path = image_path
        dataset = gdal.OpenShared(image_path, gdal.GA_ReadOnly)
        band = dataset.GetRasterBand(1)
        self.no_data = band.GetNoDataValue()
        self.rows, self.cols = dataset.RasterYSize, dataset.RasterXSize
        self.backward_mask = self._ignore_backward(image_path)
        self.im_width = dataset.RasterXSize # 影像宽, W
        self.im_height = dataset.RasterYSize # 影像高, H
        self.im_bands = dataset.RasterCount # 波段数
        dataset = None
        if multith_mode:
            self.dataset = image_path
        else:
            self.dataset = gdal.Open(image_path, gdal.GA_ReadOnly)

    def _ignore_backward(self, image_path):
        '''分块计算背景掩膜值, 默认分块大小为512'''
        dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
        block_size = 512
        if self.cols> (2 * block_size) and self.rows > (2 * block_size):
            pass
        else:
            block_size = min(self.rows, self.cols) # 如果行列都较小, 则使用行列最小值作为分块大小
        mask = np.empty((self.rows, self.cols), dtype=bool)
        for i in range(0, self.rows, block_size):
            for j in range(0, self.cols, block_size):
                # 计算当前块的实际高度和宽度(避免越界)
                actual_rows = min(block_size, self.rows - i)
                actual_cols = min(block_size, self.cols - j)
                # 读取当前块的所有波段数据(形状: [bands, actual_rows, actual_cols])
                block_data = dataset.ReadAsArray(xoff=j, yoff=i, xsize=actual_cols, ysize=actual_rows)
                block_mask = np.all(block_data == self.no_data, axis=0)
                mask[i:i + actual_rows, j:j + actual_cols] = ~block_mask
        dataset = None
        return mask
    
    def __del__(self):
        """释放GDAL资源"""
        if hasattr(self, 'dataset') and self.dataset:
            self.dataset = None

class MultiImageRandomBlockSampler(object):
    """多图像分块随机裁剪采样器, 用于从多个图像中分区域随机选择图像块"""
    def __init__(self, image_paths_list, patch_size, multith_mode=False):
        self.image_paths_list = image_paths_list
        print('Calculating data area, collecting information...')
        self.im_info_objs = [Im_info(image_path, multith_mode) for image_path in image_paths_list]
        self.patch_size = patch_size
        self.idx_list = [] # [(Im_info: np.array([0,2,3,...])),...]
        self.out_list = [] # [(Im_info: 1), (Im_info: 98)...]
        self.iters = 0

        self.cal_area()
        self.reset()
    
    def cal_area(self):
        """计算每个数据的分块区域"""
        image_block = self.patch_size
        for obj in self.im_info_objs:
            mask = obj.backward_mask
            nums = mask.size
            rows, cols = mask.shape
            idx_matrix = np.arange(nums).reshape(mask.shape)
            for i in range(0, rows, image_block):
                for j in range(0, cols, image_block):
                    # 计算当前块的实际高度和宽度(避免越界)
                    actual_rows = min(image_block, rows - i)  # 实际高
                    actual_cols = min(image_block, cols - j)  # 实际宽
                    block_idx = idx_matrix[i:i + actual_rows, j:j+actual_cols].reshape(-1)
                    mask_idx = mask[i:i + actual_rows, j:j+actual_cols].reshape(-1)
                    if np.any(mask_idx):
                        block_idx = block_idx[mask_idx]
                        self.idx_list.append((obj, block_idx))
                    else:
                        continue
        self.iters = len(self.idx_list)
    
    def reset(self):
        self.out_list = [(i[0], random.choice(i[1])) for i in self.idx_list]

class MIRBS_Dataset(Dataset):
    """多图像动态分块随机裁剪数据集, 使用该类时请设置numworkers=0(经过实验, 预先读取gdal数据会使得
    加载数据更快, 比多进程更快)。不适用多进程, 应该也不适用分布式训练, 如有需要, 做一些适当的修改即可"""
    def __init__(self, 
                 image_paths_list, # 图像路径列表
                 patch_size,
                 multith_mode = False):
        self.MIRBS = MultiImageRandomBlockSampler(image_paths_list, patch_size, multith_mode=multith_mode)
        self.patch_size = patch_size
        self.left_top = int(patch_size / 2 - 1) if patch_size % 2 == 0 else int(patch_size // 2)
        self.right_bottom = int(patch_size / 2) if patch_size % 2 == 0 else int(patch_size // 2)
        self.list = self.MIRBS.out_list
        self.multith_mode = multith_mode
        image = self.__getitem__(0)
        self.data_shape = image.shape
        self.scale = 1e-4 if (image.abs().max() > 100) else 1.0
    
    def reset(self): # 重置索引,外部调用
        self.MIRBS.reset()
        self.list = self.MIRBS.out_list

    def __len__(self):
        return self.MIRBS.iters
    
    def __getitem__(self, idx):
        obj, index = self.list[idx]
        im_width = obj.im_width
        im_height = obj.im_height
        im_bands = obj.im_bands
        if self.multith_mode:
            dataset = gdal.Open(obj.image_path, gdal.GA_ReadOnly) # 多进程模式下每次都重新打开dataset
        else:
            dataset = obj.dataset
        row = index // im_width # 计算行索引
        col = index % im_width # 计算列索引

        # 计算裁剪窗口
        col_start = col - self.left_top
        row_start = row - self.left_top
        col_end = col + self.right_bottom + 1
        row_end = row + self.right_bottom + 1
        
        # 计算实际可读取范围
        read_col = max(0, col_start)
        read_row = max(0, row_start)
        read_width = min(col_end, im_width) - read_col
        read_height = min(row_end, im_height) - read_row
        
        # 创建并填充数组
        block = np.full((im_bands, self.patch_size, self.patch_size), 
                           0, dtype=np.float32)
        if read_width > 0 and read_height > 0:
            data = dataset.ReadAsArray(read_col, read_row, read_width, read_height)
            offset_x = read_col - col_start
            offset_y = read_row - row_start
            block[:, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
        scale = self.scale if self.scale is not None else 1.0
        block = torch.tensor(block, dtype=torch.float32) * scale
        if self.patch_size == 1:
            block = block.squeeze()
        return block
