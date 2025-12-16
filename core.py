try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
from typing import Generator
import numpy as np
from gdal_utils import write_data_to_tif, point_shp_to_mask, mask_to_point_shp, mutipoint_shp_to_mask, Mianvector2mask, sieve_filtering
from algorithms import pca, noise_estimation, MNF

class Hyperspectral_Image:
    def __init__(self, input:str | None = None, init_fig: bool = False):
        self.dataset, self.rows, self.cols, self.bands = None, None, None, None
        self.no_data = None
        self.sampling_position = None # 二维矩阵，标记了样本的取样点和类别信息
        self.backward_mask = None # [rows, cols] 背景掩膜
        self.ori_img = None # [rows, cols, 3] 拉伸影像
        self.enhance_data = None # [rows, cols, bands] 增强原始数据
        self.enhance_img = None #[rows, cols, 3] 增强-拉伸影像
        self.geotransform = None
        self.projection = None
        if input is not None:
            self.init(input, init_fig=init_fig)

    def __del__(self):
        self.dataset = None # 释放内存

    def init(self, filepath: str, init_fig: bool = True, rgb: tuple[int, int, int] = (1,2,3)) -> int:
        try:
            dataset = gdal.Open(filepath)
            bands = dataset.RasterCount
            rows, cols = dataset.RasterYSize, dataset.RasterXSize
            self.dataset, self.rows, self.cols, self.bands = dataset, rows, cols, bands
            self.geotransform = dataset.GetGeoTransform()
            self.projection = dataset.GetProjection()
            band = self.dataset.GetRasterBand(1)
            self.no_data = band.GetNoDataValue()
            band = None
            if init_fig: # 根据需要加载影像数据
                self.init_fig_data(rgb=rgb)
            return 0 # 代表数据导入成功
        except (AttributeError,RuntimeError):
            return 1

    def create_vector(self, mask: np.ndarray, out_file: str) -> None: # mask 转单矢量点
        mask_to_point_shp(mask, self.dataset, out_file)
        
    def create_mask(self, input_file: str) -> np.ndarray: # 矢量点转mask，点的数值由“class”字段确定
        return point_shp_to_mask(input_file, self.dataset)
    
    def create_mask_from_mutivector(self, inputdir: str) -> np.ndarray: # 多矢量点转mask
        return mutipoint_shp_to_mask(inputdir, self.dataset)
    
    def sieve_filtering(self, output_tif_path: str, threshold_pixels: int, connectedness: int =8) -> None:
        '''使用GDAL的SieveFilter去除碎斑'''
        if self.backward_mask is None:
            self.backward_mask = self.ignore_backward()
        sieve_filtering(self.dataset, output_tif_path, threshold_pixels, connectedness, mask=self.backward_mask)

    def save_tif(self, filename: str, img_data: np.ndarray, nodata: float | int | None = None, mask: np.ndarray | None = None) -> bool:
        '''将(rows, cols,  bands)或(rows, cols)的数据存为tif格式, tif具有与img相同的投影信息'''
        nodata = self.no_data if nodata is None else nodata
        if len(img_data.shape) == 3:
            write_data_to_tif(filename, img_data.transpose(2,0,1), self.geotransform, self.projection,
                          nodata_value=nodata, mask=mask)
        elif len(img_data.shape) == 2:
            write_data_to_tif(filename, img_data, self.geotransform, self.projection,
                nodata_value=nodata, mask=mask)
        else:
            raise ValueError("The input dims must be 2 or 3")
        return True

    def init_fig_data(self, rgb: tuple[int, int, int] = (1,2,3)): # 计算背景掩膜，生成拉伸图像
        band = self.dataset.GetRasterBand(1)
        self.backward_mask = self.ignore_backward()  # 初始化有效像元位置
        r,g,b = rgb
        self.compose_rgb(r=r, g=g, b=b)
    
    def Mianvector2raster(self, input_shp: str, out_tif: str, nodata: float|int = 0, fill_value: int = 255) -> None:
        '''将矢量面转为栅格，矢量内区域设为fill_value，外部区域设为0，处理结果与原始数据大小一致'''
        mask,_,_ = Mianvector2mask(vector_path=input_shp, tif_path=self.dataset, fill_value=fill_value)
        mask = mask.astype(np.uint8)
        self.save_tif(out_tif, mask, nodata=nodata)
    
    def Mianvector_clip_tif(self, input_shp: str, out_tif: str, nodata: float|int = 0, fill_value: int = 1) -> None:
        '''根据面shp文件对影像进行裁剪，矢量内区域保留，外部区域设为nodata，处理结果与原始数据大小一致'''
        mask,_,_ = Mianvector2mask(vector_path=input_shp, tif_path=self.dataset, fill_value=fill_value)
        mask = mask.astype(bool)
        self.save_tif(out_tif, self.get_dataset(), nodata=nodata, mask=mask) # 保存裁剪影像，原始数据不缩放

    def update(self,r: int, g: int, b: int, show_enhance_img: bool = False): # 根据所选择rgb组合更新拉伸图像
        if show_enhance_img:
            self.compose_enhance(r,g,b)
        else:
            self.compose_rgb(r,g,b)

    def compose_rgb(self, r: int, g: int, b: int, stretch: bool = True) -> np.ndarray: # 合成彩色图像
        try:
            r_band = self.get_band_data(r)
            g_band = self.get_band_data(g)
            b_band = self.get_band_data(b)
        except:
            print("波段序号无效, 波段序号最小为1! 将默认使用第1个波段合成影像")
            r_band = self.get_band_data(1)
            g_band = self.get_band_data(1)
            b_band = self.get_band_data(1)
        try:# 拉伸出错可能是mask全为False，忽略
            if stretch:
                r_band = linear_2_percent_stretch(r_band, self.backward_mask)
                g_band = linear_2_percent_stretch(g_band, self.backward_mask)
                b_band = linear_2_percent_stretch(b_band, self.backward_mask)
        except ValueError as e:
            print(f'Error in linear stretch: {e}')
            r_band = r_band[self.backward_mask]
            g_band = g_band[self.backward_mask]
            b_band = b_band[self.backward_mask]
        rgb = np.dstack([r_band, g_band, b_band]).squeeze().astype(np.float32)
        self.ori_img = np.zeros((self.rows, self.cols, 3)) + 1
        self.ori_img[self.backward_mask] = rgb
        return self.ori_img

    def compose_enhance(self, r: int, g: int, b: int, stretch: bool = True) -> None: # 合成增强彩色图像
        '''这里为了和tif波段组合统一，读取enhance_data波段数据，波段减一'''
        r_band = self.enhance_data[:, :, r-1]
        g_band = self.enhance_data[:, :, g-1]
        b_band = self.enhance_data[:, :, b-1]
        try:
            if stretch:
                r_band = linear_2_percent_stretch(r_band, self.backward_mask)
                g_band = linear_2_percent_stretch(g_band, self.backward_mask)
                b_band = linear_2_percent_stretch(b_band, self.backward_mask)
        except ValueError as e:
            print(f'Error in linear stretch: {e}')
            r_band = r_band[self.backward_mask]
            g_band = g_band[self.backward_mask]
            b_band = b_band[self.backward_mask]
        rgb = np.dstack([b_band, g_band, r_band]).squeeze().astype(np.float32)
        self.enhance_img = np.zeros((self.rows, self.cols, 3)) + 1
        self.enhance_img[self.backward_mask] = rgb

    def get_band_data(self, band_idx: int) -> np.ndarray:
        """获取指定波段的数据
        :return (rows, cols)"""
        band = self.dataset.GetRasterBand(band_idx)
        band_data = band.ReadAsArray()
        return band_data

    def get_dataset(self) -> np.ndarray:
        '''return: (bands, rows, cols)的numpy数组，数据类型为float32'''
        dataset = self.dataset.ReadAsArray()
        return dataset

    def ignore_backward(self) -> np.ndarray:
        '''分块计算背景掩膜值，默认分块大小为512'''
        print("Calculating The Whole Background Mask...")
        no_data = self.no_data if self.no_data is not None else 0 # 如果原始数据没有nodata值，默认为0
        if self.no_data is None:
            print("Warning: The input data has no NoData value, default NoData=0 is used!")
        block_size = 512 if self.cols> (2 * 512) and self.rows > (2 * 512) else min(self.rows, self.cols)
        mask = np.empty((self.rows, self.cols), dtype=bool)
        for i in range(0, self.rows, block_size):
            for j in range(0, self.cols, block_size):
                # 计算当前块的实际高度和宽度（避免越界）
                actual_rows = min(block_size, self.rows - i)
                actual_cols = min(block_size, self.cols - j)
                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
                block_data = self.dataset.ReadAsArray(xoff=j, yoff=i, xsize=actual_cols, ysize=actual_rows)
                block_mask = np.all(block_data == no_data, axis=0)
                mask[i:i + actual_rows, j:j + actual_cols] = ~block_mask
        return mask

    def image_enhance(self, f: str = 'PCA', 
                      n_components: int = 10, 
                      row_slice: tuple[int, int] | None=None, 
                      col_slice: tuple[int, int] | None=None, 
                      band_slice: tuple[int, int] | None=None) -> np.ndarray:
        # 影像增强
        def to_slice(s=None):
            """s:(int, int) or int or None"""
            if s is None:
                return slice(None)
            else:return slice(*s) if isinstance(s, tuple) else s
        no_data = self.no_data if self.no_data is not None else 0 # 如果原始数据没有nodata值，默认为0
        if self.no_data is None:
            print("Warning: The input data has no NoData value, default NoData=0 is used!")
        ori_dataset = self.get_dataset().transpose(1, 2, 0)
        dataset = ori_dataset[self.backward_mask]
        if f == 'PCA':
            dataset = pca(dataset, n_components=n_components)
        elif f == 'MNF':
            if row_slice is None and col_slice is None and band_slice is None:
                mask = self.backward_mask.astype(np.int16)
            else: mask = None
            row_slice, col_slice, band_slice = to_slice(row_slice), to_slice(col_slice), to_slice(band_slice)
            noise_stats = noise_estimation(ori_dataset[row_slice, col_slice, band_slice], mask=mask)
            dataset = MNF(dataset, noise_stats, n_components)
        self.enhance_data = np.full((self.rows, self.cols, n_components), no_data, dtype=np.float32)
        self.enhance_data[self.backward_mask] = dataset
        self.compose_enhance(1,2,3)
        return self.enhance_img

    def generate_sampling_mask(self, sample_fraction: float = 0.001) -> np.ndarray:
        """经过该函数进行随机取样，取样位置的标签为1"""
        rows, cols, bands = self.rows, self.cols, self.bands
        mask = self.backward_mask if self.backward_mask is not None else self.ignore_backward()

        total_pixels = rows * cols
        num_samples = int(total_pixels * sample_fraction)
        all_indices = np.arange(total_pixels)
        sampled_indices = np.random.choice(all_indices, size=num_samples, replace=False)
        sampled_rows = sampled_indices // cols
        sampled_cols = sampled_indices % cols
        sampling_position = np.zeros((rows, cols), dtype=np.uint8)
        sampling_position[sampled_rows, sampled_cols] = 1
        sampling_position[~mask] = 0
        return sampling_position

    def image_block_iter(self, block_size: int = 256, 
                         patch_size: int = 30) -> Generator[tuple[np.ndarray, np.ndarray, int, int], None, None]: # 该迭代器用于预测大影像
        """迭代器，返回分块数据和块的左上角坐标"""
        left_top = int(patch_size / 2 - 1) if patch_size % 2 == 0 else int(patch_size // 2)
        right_bottom = int(patch_size / 2) if patch_size % 2 == 0 else int(patch_size // 2)
        for i in range(0, self.rows, block_size):
            for j in range(0, self.cols, block_size):
                # 计算当前块的实际高度和宽度（避免越界）
                actual_rows = min(block_size + patch_size - 1, self.rows - i)  # 实际高
                actual_cols = min(block_size + patch_size - 1, self.cols - j)  # 实际宽
                xoff = 0 if (j - left_top) < 0 else j - left_top
                left_pad = left_top if (j - left_top) < 0 else 0
                yoff = 0 if (i - left_top) < 0 else i - left_top
                top_pad = left_top if (i - left_top) < 0 else 0
                if (j - left_top) < 0: actual_cols -= left_top
                if (i - left_top) < 0: actual_rows -= left_top
                # 计算边缘pad
                if actual_cols == (self.cols - j): # 如果实际宽度已经接近了最右边界
                    pad = actual_cols - block_size
                    right_pad = right_bottom - pad if pad >=0 else right_bottom
                    actual_cols += left_top
                else:
                    right_pad = 0
                if actual_rows == (self.rows - i):
                    pad = actual_rows - block_size
                    bottom_pad = right_bottom - pad if pad >=0 else right_bottom
                    actual_rows += left_top
                else:
                    bottom_pad = 0
                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
                block_data = self.dataset.ReadAsArray(xoff=xoff, yoff=yoff, xsize=actual_cols, ysize=actual_rows)
                block_data = np.pad(block_data, [(0, 0), (top_pad, bottom_pad), (left_pad, right_pad)], 'constant')
                # 经过上面的计算位于左上区域和中间区域的块大小一律为（image_block + block_size - 1，image_block + block_size - 1）
                # 比如如果参数是64， 17， 那么裁剪的块大小为（80, 80）
                row_block = min(block_size, self.rows - i) # 记录真实窗口大小
                col_block = min(block_size, self.cols - j)
                block_sampling_mask = self.backward_mask[i:i + row_block, j:j + col_block]
                yield block_data, block_sampling_mask, i, j
    
    def read_dataset_from_mask(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]: # 根据mask读取数据，形成数据集, 在机器学习中使用
        """大影像读取数据集"""
        mask = mask.astype(np.int16)
        index_list = []
        label_list = []
        for y in range(self.rows):
            for x in range(self.cols):
                mask_value = mask[y, x]
                if mask_value != 0:  # 0表示背景，跳过
                    index_list.append((x, y))
                    label_list.append(mask_value - 1)  # 标签从0开始
        label = np.array(label_list, dtype=np.int16)
        num_samples = len(label_list)

        data = np.zeros((num_samples, self.bands), dtype=np.float32)
        for i, (x, y) in enumerate(index_list):
            pixel_data = self.dataset.ReadAsArray(xoff=x, yoff=y, xsize=1, ysize=1).flatten()
            data[i, :] = pixel_data
        return data, label
    
def linear_2_percent_stretch(band_data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    '''
    线性拉伸
    :param band_data: 单波段数据[rows, cols]
    :param mask: [rows, cols], bool类型，True表示有效像元
    :return: stretched_band[valid_pixels,]
    '''
    if mask is not None and np.sum(mask) == 0:
        return np.zeros_like(band_data)
    band_data = band_data[mask] if mask is not None else band_data.reshape(band_data.shape[0]*band_data.shape[1])
    # 计算2%和98%分位数
    lower_percentile = np.percentile(band_data, 2)
    upper_percentile = np.percentile(band_data, 98)
    if lower_percentile == upper_percentile:
        return np.zeros_like(band_data)
    # 拉伸公式：将数值缩放到 0-1 范围内
    stretched_band = np.clip((band_data - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)
    return stretched_band

def linear_percent_stretch(band_data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None and np.sum(mask) == 0:
        return np.zeros_like(band_data)
    band_data = band_data[mask] if mask is not None else band_data.reshape(band_data.shape[0]*band_data.shape[1])
    # 计算2%和98%分位数
    lower_percentile = np.min(band_data)
    upper_percentile = np.max(band_data)
    if lower_percentile == upper_percentile:
        return np.zeros_like(band_data)
    # 拉伸公式：将数值缩放到 0-1 范围内
    stretched_band = np.clip((band_data - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)
    return stretched_band

if __name__ == '__main__':
    x = np.random.rand(100, 10)
    slice = slice(None)
    y = x[slice]
    print(y.shape)