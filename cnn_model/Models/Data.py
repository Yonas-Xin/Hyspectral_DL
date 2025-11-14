import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import h5pickle
except:
    pass

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
def read_tif_with_gdal(tif_path):
    '''读取栅格原始数据
    返回dataset[bands,H,W]'''
    dataset = gdal.Open(tif_path)
    dataset = dataset.ReadAsArray()
    if dataset.dtype == np.int16:
        dataset = dataset.astype(np.float32) * 1e-4
    return dataset
class CNN_Dataset(Dataset):
    '''用于3D——CNN训练和测试'''
    def __init__(self, data_list, transform=None):
        """
        将列表划分为数据集
        Args:
            transform (callable, optional): 图像变换操作
        """
        self.image_paths = data_list
        self.transform = transform
        image, label = self.__getitem__(0)
        self.data_shape = image.shape  # 获取数据形状

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        """
        根据索引返回图像及其标签
        image（1，C，H，W）
        """
        image_path, label = self.image_paths[idx].split()
        label = np.array(label, dtype=np.int16)
        image = read_tif_with_gdal(image_path)
        bands, h, w = image.shape
        if h == 1 and w == 1:  # 自动识别1D 输入
            image = image.squeeze() # (bands,)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        image = torch.tensor(image, dtype=torch.float32)
        return image, label

class Predict_Dataset(Dataset):
    '''构造用于3D编码器的输入''' 
    def __init__(self, patch_size):
        self.block_size = patch_size
        self.data = None
        self.rows, self.cols = 0, 0
        self.idx = None

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
        block = torch.from_numpy(block).float()
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

        mask = background_mask.astype(np.bool)
        self.idx = self.idx.reshape(self.rows, self.cols)
        self.idx = self.idx[mask]
    
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