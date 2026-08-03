try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import os
from typing import Generator
import numpy as np
from gdal_utils import write_data_to_tif, point_shp_to_mask, subset_image, \
    mask_to_point_shp, mutipoint_shp_to_mask, Mianvector2mask, sieve_filtering

def GET_WATER_INDICES() -> list[tuple[int, int]]:
    """获取水汽吸收带的波段范围"""
    return [(1350, 1450), (1800, 1950)]


def _read_envi_sidecar_hdr(filepath: str) -> dict[str, str]:
    """Read metadata from a sibling ENVI .hdr file when GDAL does not expose it."""
    candidates = [
        os.path.splitext(filepath)[0] + '.hdr',
        filepath + '.hdr',
    ]
    hdr_path = next((path for path in candidates if os.path.exists(path)), None)
    if hdr_path is None:
        return {}

    metadata = {}
    current_key = None
    current_value_parts = []
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                text = line.strip()
                if not text or text.upper() == 'ENVI':
                    continue

                if current_key is not None:
                    current_value_parts.append(text)
                    if '}' in text:
                        metadata[current_key] = ' '.join(current_value_parts)
                        current_key = None
                        current_value_parts = []
                    continue

                if '=' not in text:
                    continue
                key, value = text.split('=', 1)
                key = key.lower().strip()
                value = value.strip()
                if value.startswith('{') and '}' not in value:
                    current_key = key
                    current_value_parts = [value]
                else:
                    metadata[key] = value
    except OSError:
        return {}

    return metadata


class Hyperspectral_Image:
    def __init__(self, input:str | None = None, init_fig: bool = False):
        self.dataset, self.rows, self.cols, self.bands = None, None, None, None
        self.no_data = None
        self.backward_mask = None # [rows, cols] 背景掩膜
        self.ori_img = None # [rows, cols, 3] 拉伸影像
        self.enhance_data = None # [rows, cols, bands] 增强原始数据
        self.enhance_img = None #[rows, cols, 3] 增强-拉伸影像
        self.geotransform = None
        self.projection = None
        self.metadata = {}  # 元数据字典（wavelength, fwhm 等）
        self.wavelengths = None  # np.ndarray, 各波段中心波长 (nm)
        self.fwhm = None  # np.ndarray, 各波段半高宽 (nm)
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
            # 读取元数据（波长、FWHM 等）
            if init_fig: # 根据需要加载影像数据
                self.init_fig_data(rgb=rgb)
            self.metadata = {}
            for domain in [None, 'ENVI']:
                meta = dataset.GetMetadata(domain) if domain else dataset.GetMetadata()
                if meta:
                    for key, value in meta.items():
                        key_lower = key.lower().strip()
                        if key_lower not in self.metadata:
                            self.metadata[key_lower] = value
            # 重点解析 wavelength 和 fwhm
            sidecar_metadata = _read_envi_sidecar_hdr(filepath)
            for key, value in sidecar_metadata.items():
                if key not in self.metadata:
                    self.metadata[key] = value
            self.wavelengths = self._parse_metadata_array('wavelength')
            self.fwhm = self._parse_metadata_array('fwhm')
            if self.wavelengths is not None:
                print(f"INFO: 读取到 {len(self.wavelengths)} 个波段的波长信息 ({self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm)")
            if self.fwhm is not None:
                print(f"INFO: 读取到 {len(self.fwhm)} 个波段的 FWHM 信息")
            if self.no_data is None:
                print("Warning: The input data has no NoData value, default NoData=0 is used!")
                self.no_data = 0 if self.no_data is None else self.no_data

            return 0 # 代表数据导入成功
        except (AttributeError,RuntimeError):
            return 1

    def _parse_metadata_array(self, key: str) -> np.ndarray | None:
        """从 self.metadata 中解析指定 key 的逗号分隔列表为 numpy 数组"""
        value = self.metadata.get(key)
        if value is None:
            return None
        raw = value.strip().strip('{}')
        try:
            arr = np.fromstring(raw, sep=',')
            return arr if len(arr) > 0 else None
        except ValueError:
            return None

    @staticmethod
    def _detect_overlap_mask(wavelengths: np.ndarray) -> np.ndarray:
        """
        检查波长数组是否单调递增且无重复值
        """
        wavelengths_copy = wavelengths.copy()
        diff = np.diff(wavelengths_copy)
        index = 0
        count = 0
        while np.any(diff <= 0):
            index = np.where(diff <= 0)[0][0]
            # 计算重合波段的上下限
            low_bound = wavelengths_copy[index + 1]
            upper_bound = wavelengths_copy[index]
            repeat_mask = (wavelengths_copy >= low_bound) & (wavelengths_copy <= upper_bound)
            overlapping_wavelengths = wavelengths_copy[repeat_mask]
            if count == 0:
                print(f"Found overlapping wavelengths:\n {overlapping_wavelengths}")
            upper_index = np.where(overlapping_wavelengths == upper_bound)[0]
            if (overlapping_wavelengths.size - upper_index) > 2:
                wavelengths_copy = np.delete(wavelengths_copy, index + 1)
            else:
                wavelengths_copy = np.delete(wavelengths_copy, index)
            diff = np.diff(wavelengths_copy)
            count += 1
        if index == 0:
            return np.zeros(len(wavelengths), dtype=bool)
        slice_num = (index, index+count)
        overlap_mask = np.zeros(len(wavelengths), dtype=bool)
        for i in range(slice_num[0]-1, slice_num[1]+1):
            overlap_mask[i] = True
            print(f"INFO: The deleted overlapping band {i+1} ({wavelengths[i]} nm)")
        return overlap_mask

    @staticmethod
    def _detect_overlap_mask_copy(wavelengths: np.ndarray) -> np.ndarray:
        """
        检查波长数组是否单调递增且无重复值
        """
        wavelengths_copy = wavelengths.copy()
        diff = np.diff(wavelengths_copy)
        index = 0
        count = 0
        index_list = []
        if_max_overlap_value = True
        while np.any(diff <= 0):
            index = np.where(diff <= 0)[0][0]
            if if_max_overlap_value:
                delete_index = index
                if_max_overlap_value = False
            else:
                delete_index = index + 1
            low_bound = wavelengths_copy[index + 1]
            upper_bound = wavelengths_copy[index]
            repeat_mask = (wavelengths_copy >= low_bound) & (wavelengths_copy <= upper_bound)
            overlapping_wavelengths = wavelengths_copy[repeat_mask]
            if count == 0:
                print(f"Found overlapping wavelengths:\n {overlapping_wavelengths}")
            index_list.append(delete_index + count)
            wavelengths_copy = np.delete(wavelengths_copy, delete_index)
            diff = np.diff(wavelengths_copy)
            count += 1
        if len(index_list) == 0:
            return np.zeros(len(wavelengths), dtype=bool)
        overlap_mask = np.zeros(len(wavelengths), dtype=bool)
        for i in index_list:
            overlap_mask[i] = True
            print(f"INFO: The deleted overlapping band {i+1} ({wavelengths[i]} nm)")
        return overlap_mask

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

    def save_tif(self, filename: str, img_data: np.ndarray, nodata: float | int | None = None, 
                    mask: np.ndarray | None = None, write_metadata: bool = False) -> bool:
        '''将(rows, cols,  bands)或(rows, cols)的数据存为tif格式, tif具有与img相同的投影信息'''
        nodata = self.no_data if nodata is None else nodata
        meta = self.metadata if write_metadata else None
        if len(img_data.shape) == 3:
            write_data_to_tif(filename, img_data.transpose(2,0,1), self.geotransform, self.projection,
                          nodata_value=nodata, mask=mask, metadata=meta)
        elif len(img_data.shape) == 2:
            write_data_to_tif(filename, img_data, self.geotransform, self.projection,
                nodata_value=nodata, mask=mask)
        else:
            raise ValueError("The input dims must be 2 or 3")
        return True
    
    def subset_image_from_wavelength(self, 
                    output_path: str,
                    water_vapor_ranges: list[tuple[float, float]] | None = None,
                    remove_overlap: bool = True) -> None:
        """
        根据水汽吸收波段范围（和可选的重合波段检测）去除对应波段，
        将保留的波段子集写出到 output_path。
        :param output_path: 输出影像路径
        :param water_vapor_ranges: 水汽吸收波段范围列表 [(min_nm, max_nm), ...]，
                                   默认使用 GET_WATER_INDICES() 返回的范围
        :param remove_overlap: 是否同时去除 VNIR-SWIR 重合波段（默认 True）
        """
        if water_vapor_ranges is None:
            water_vapor_ranges = GET_WATER_INDICES()
        wavelengths = self.wavelengths
        n_bands = self.bands
        delete_mask = np.zeros(n_bands, dtype=bool)
        for wv_min, wv_max in water_vapor_ranges:
            delete_mask |= (wavelengths >= wv_min) & (wavelengths <= wv_max)
        wv_count = int(np.sum(delete_mask))
        print(f"水汽波段 (共 {wv_count} 个):")
        for rng in water_vapor_ranges:
            in_range = (wavelengths >= rng[0]) & (wavelengths <= rng[1])
            n = int(np.sum(in_range))
            if n > 0:
                print(f"  {rng[0]:.0f}-{rng[1]:.0f} nm: {n} 个波段")
        overlap_count = 0
        if remove_overlap:
            overlap_mask = self._detect_overlap_mask_copy(wavelengths)
            delete_mask |= overlap_mask
            overlap_count = int(np.sum(overlap_mask))
        total_deleted = int(np.sum(delete_mask))
        remaining = n_bands - total_deleted
        print(f"\n总计删除: {total_deleted} 个波段 (水汽: {wv_count}, 重合: {overlap_count})")
        print(f"保留波段: {remaining} 个")
        if remaining == 0:
            raise ValueError("所有波段都被标记为删除，请检查参数设置！")
        keep_indices = (np.where(~delete_mask)[0] + 1).tolist()
        updated_metadata = self.metadata.copy()
        keep_mask = ~delete_mask
        metadata_arrays = {
            'wavelength': self.wavelengths,
            'fwhm': self.fwhm,
        }
        for key, fallback_arr in metadata_arrays.items():
            arr = self._parse_metadata_array(key)
            if arr is None and fallback_arr is not None:
                arr = np.asarray(fallback_arr)
            if arr is not None and len(arr) == n_bands:
                kept = arr[keep_mask]
                updated_metadata[key] = '{' + ', '.join(f'{v:.6f}' for v in kept) + '}'
        if 'wavelength units' not in updated_metadata:
            updated_metadata['wavelength units'] = 'Nanometers'
        subset_image(self.dataset, output_path=output_path,
                     band_indices=keep_indices, metadata=updated_metadata)

    def init_fig_data(self, rgb: tuple[int, int, int] = (1,2,3)): # 计算背景掩膜，生成拉伸图像
        band = self.dataset.GetRasterBand(1)
        self.backward_mask = self.ignore_backward()  # 初始化有效像元位置
        r,g,b = rgb
        self.ori_img = self._compose_rgb(r=r, g=g, b=b)
    
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
            self.enhance_img = self._compose_enhance(r,g,b)
        else:
            self.ori_img = self._compose_rgb(r,g,b)

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
        block_size = 512 if self.cols> (2 * 512) and self.rows > (2 * 512) else min(self.rows, self.cols)
        mask = np.empty((self.rows, self.cols), dtype=bool)
        for i in range(0, self.rows, block_size):
            for j in range(0, self.cols, block_size):
                # 计算当前块的实际高度和宽度（避免越界）
                actual_rows = min(block_size, self.rows - i)
                actual_cols = min(block_size, self.cols - j)
                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
                block_data = self.dataset.ReadAsArray(xoff=j, yoff=i, xsize=actual_cols, ysize=actual_rows)
                block_mask = np.all(block_data == self.no_data, axis=0)
                mask[i:i + actual_rows, j:j + actual_cols] = ~block_mask
        print("Background Mask Calculation Completed.")
        return mask
    
    def image_enhance(self, f: str = 'PCA', 
                      n_components: int = 10, 
                      row_slice: tuple[int, int] | None=None, 
                      col_slice: tuple[int, int] | None=None, 
                      band_slice: tuple[int, int] | None=None,
                      block_size: int = 512) -> np.ndarray:
        if self.backward_mask is None:
            self.backward_mask = self.ignore_backward()
        block_size = self._resolve_block_size(block_size, self.bands)
        valid_bands_mask = self._compute_valid_bands_mask(self.no_data, block_size)
        if not np.all(valid_bands_mask):
            removed_count = np.sum(~valid_bands_mask)
            print(f"Warning: Removed {removed_count} bands containing only NaN or NoData.")
        band_indices = np.where(valid_bands_mask)[0]
        if band_indices.size == 0:
            raise ValueError("No valid bands found for enhancement.")
        if band_slice is not None:
            band_slice = self._to_slice(band_slice)
            band_indices = band_indices[band_slice]
        if band_indices.size == 0:
            raise ValueError("No valid bands found after band_slice.")
        n_components = min(n_components, band_indices.size)
        if f == 'PCA':
            mean, cov = self._compute_mean_cov(band_indices, block_size)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvectors_sorted = eigenvectors[:, sorted_idx]
            eigenvectors_selected = -eigenvectors_sorted[:, :n_components]
            self._apply_linear_transform(eigenvectors_selected, mean, band_indices, block_size, self.no_data)
        elif f == 'MNF':
            use_mask = row_slice is None and col_slice is None and band_slice is None
            row_slice = self._to_slice(row_slice)
            col_slice = self._to_slice(col_slice)
            data_size_gb = self._estimate_data_size_gb()
            if data_size_gb < 2.0:
                print(f"Info: Data size ({data_size_gb:.2f} GB) is less than 2 GB, computing MNF directly in memory without chunked processing.")
                mean, signal_cov, noise_cov, transform = self._compute_mnf_direct(band_indices, n_components, row_slice, col_slice, use_mask)
            else:
                print(f"Info: Data size ({data_size_gb:.2f} GB) >= 2 GB, using chunked block-based MNF processing.")
                noise_cov = self._compute_noise_cov(band_indices, block_size, row_slice, col_slice, use_mask)
                mean, signal_cov = self._compute_mean_cov(band_indices, block_size)
                noise_eigvals, noise_eigvecs = np.linalg.eigh(noise_cov)
                noise_eigvals = np.clip(noise_eigvals, a_min=1e-12, a_max=None)
                noise_inv_sqrt = noise_eigvecs @ np.diag(1.0 / np.sqrt(noise_eigvals)) @ noise_eigvecs.T
                whitened_signal = noise_inv_sqrt @ signal_cov @ noise_inv_sqrt.T
                signal_eigvals, signal_eigvecs = np.linalg.eigh(whitened_signal)
                sorted_idx = np.argsort(signal_eigvals)[::-1]
                signal_eigvecs = signal_eigvecs[:, sorted_idx]
                transform = noise_inv_sqrt @ signal_eigvecs[:, :n_components]
                self._apply_linear_transform(transform, mean, band_indices, block_size, self.no_data)
        else:
            raise ValueError(f"Unsupported enhance method: {f}")
        self.enhance_img = self._compose_enhance(1,2,3)
        return self.enhance_data, self.enhance_img

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
                if (j - left_top) < 0 and self.cols-actual_cols>0: actual_cols -= left_top # 位于左边界
                if (i - left_top) < 0 and self.rows-actual_rows>0: actual_rows -= left_top # 位于上边界
                # 计算边缘pad
                if actual_cols == (self.cols - j): # 如果实际宽度已经接近了最右边界
                    pad = actual_cols - block_size
                    right_pad = right_bottom - pad if pad >=0 else right_bottom
                    if self.cols-actual_cols>0: # 如果整行都在block_size内
                        actual_cols += left_top
                else:
                    right_pad = 0
                if actual_rows == (self.rows - i) and self.rows-actual_rows>0:
                    pad = actual_rows - block_size
                    bottom_pad = right_bottom - pad if pad >=0 else right_bottom
                    if self.rows-actual_rows>0:
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
        """大影像读取数据集, 机器学习搭建数据集使用"""
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
    
    def _compose_rgb(self, r: int, g: int, b: int, stretch: bool = True) -> np.ndarray: # 合成彩色图像
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
        ori_img = np.zeros((self.rows, self.cols, 3)) + 1
        ori_img[self.backward_mask] = rgb
        return ori_img

    def _compose_enhance(self, r: int, g: int, b: int, stretch: bool = True) -> None: # 合成增强彩色图像
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
        enhance_img = np.zeros((self.rows, self.cols, 3)) + 1
        enhance_img[self.backward_mask] = rgb
        return enhance_img

    def _to_slice(self, s=None) -> slice:
        if s is None:
            return slice(None)
        return slice(*s) if isinstance(s, tuple) else s

    def _resolve_block_size(self, block_size: int, bands: int) -> int:
        if block_size <= 0:
            block_size = 256
        if bands <= 0:
            return block_size
        max_block_elements = 32_000_000
        max_side = int(np.sqrt(max_block_elements / max(1, bands)))
        if max_side <= 0:
            return block_size
        return max(32, min(block_size, max_side))

    def _estimate_data_size_gb(self) -> float:
        """估算影像数据大小（GB），基于数据集的像素数和数据类型"""
        band = self.dataset.GetRasterBand(1)
        dtype = gdal.GetDataTypeName(band.DataType)
        dtype_size_map = {
            'Byte': 1, 'UInt16': 2, 'Int16': 2,
            'UInt32': 4, 'Int32': 4, 'Float32': 4, 'Float64': 8,
            'CInt16': 4, 'CInt32': 8, 'CFloat32': 8, 'CFloat64': 16,
        }
        bytes_per_pixel = dtype_size_map.get(dtype, 4)
        total_bytes = self.rows * self.cols * self.bands * bytes_per_pixel
        return total_bytes / (1024 ** 3)

    def _compute_mnf_direct(self, 
                            band_indices: np.ndarray, 
                            n_components: int,
                            row_slice: slice | None,
                            col_slice: slice | None,
                            use_mask: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """直接在内存中计算 MNF 变换（不分块），适用于小数据集"""
        # 读取全部数据
        full_data = self.dataset.ReadAsArray()  # (bands, rows, cols)
        if full_data.ndim == 2:
            full_data = full_data[np.newaxis, :, :]
        full_data = full_data[band_indices, :, :]
        n_bands = full_data.shape[0]

        # 提取有效像元用于信号协方差
        valid_pixels = full_data[:, self.backward_mask].T.astype(np.float64)  # (N, bands)
        if valid_pixels.shape[0] < 2:
            raise ValueError("Not enough valid pixels to compute statistics.")
        mean = np.mean(valid_pixels, axis=0)
        centered = valid_pixels - mean
        signal_cov = (centered.T @ centered) / (valid_pixels.shape[0] - 1)

        # 计算噪声协方差
        row_slice = slice(None) if row_slice is None else row_slice
        col_slice = slice(None) if col_slice is None else col_slice
        row_start, row_stop, _ = row_slice.indices(self.rows)
        col_start, col_stop, _ = col_slice.indices(self.cols)
        row_end = min(row_stop, self.rows - 1)
        col_end = min(col_stop, self.cols - 1)
        sub_data = full_data[:, row_start:row_end + 1, col_start:col_end + 1]
        deltas = sub_data[:, :row_end - row_start, :col_end - col_start] - sub_data[:, 1:row_end - row_start + 1, 1:col_end - col_start + 1]
        if use_mask:
            sub_mask = self.backward_mask[row_start:row_end + 1, col_start:col_end + 1]
            valid_pairs = sub_mask[:row_end - row_start, :col_end - col_start] & sub_mask[1:row_end - row_start + 1, 1:col_end - col_start + 1]
        else:
            valid_pairs = np.ones((row_end - row_start, col_end - col_start), dtype=bool)
        if np.issubdtype(deltas.dtype, np.floating):
            valid_pairs &= np.all(np.isfinite(deltas), axis=0)
        noise_pixels = deltas[:, valid_pairs].T.astype(np.float64)  # (M, bands)
        if noise_pixels.shape[0] < 2:
            raise ValueError("Not enough samples to estimate noise statistics.")
        noise_mean = np.mean(noise_pixels, axis=0)
        noise_centered = noise_pixels - noise_mean
        noise_cov = ((noise_centered.T @ noise_centered) / (noise_pixels.shape[0] - 1)) / 2.0

        # MNF 变换
        noise_eigvals, noise_eigvecs = np.linalg.eigh(noise_cov)
        noise_eigvals = np.clip(noise_eigvals, a_min=1e-12, a_max=None)
        noise_inv_sqrt = noise_eigvecs @ np.diag(1.0 / np.sqrt(noise_eigvals)) @ noise_eigvecs.T
        whitened_signal = noise_inv_sqrt @ signal_cov @ noise_inv_sqrt.T
        signal_eigvals, signal_eigvecs = np.linalg.eigh(whitened_signal)
        sorted_idx = np.argsort(signal_eigvals)[::-1]
        signal_eigvecs = signal_eigvecs[:, sorted_idx]
        transform = noise_inv_sqrt @ signal_eigvecs[:, :n_components]

        # 直接应用变换
        self.enhance_data = np.full((self.rows, self.cols, n_components), self.no_data, dtype=np.float32)
        transformed = (valid_pixels - mean) @ transform
        self.enhance_data[self.backward_mask] = transformed.astype(np.float32)

        return mean, signal_cov, noise_cov, transform

    def _iter_blocks(self, block_size: int,
                     row_slice: slice | None = None,
                     col_slice: slice | None = None) -> Generator[tuple[int, int, np.ndarray], None, None]:
        row_slice = slice(None) if row_slice is None else row_slice
        col_slice = slice(None) if col_slice is None else col_slice
        row_start, row_stop, row_step = row_slice.indices(self.rows)
        col_start, col_stop, col_step = col_slice.indices(self.cols)
        if row_step != 1 or col_step != 1:
            raise ValueError("row_slice/col_slice step must be 1")
        for i in range(row_start, row_stop, block_size):
            for j in range(col_start, col_stop, block_size):
                actual_rows = min(block_size, row_stop - i)
                actual_cols = min(block_size, col_stop - j)
                block_data = self.dataset.ReadAsArray(xoff=j, yoff=i, xsize=actual_cols, ysize=actual_rows)
                if block_data.ndim == 2:
                    block_data = block_data[np.newaxis, :, :]
                yield i, j, block_data

    @staticmethod
    def _update_mean_cov(mean: np.ndarray | None,
                         m2: np.ndarray | None,
                         count: int,
                         batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        batch = batch.astype(np.float64, copy=False)
        batch_count = batch.shape[0]
        batch_mean = np.mean(batch, axis=0)
        centered = batch - batch_mean
        batch_m2 = centered.T @ centered
        if count == 0:
            return batch_mean, batch_m2, batch_count
        delta = batch_mean - mean
        total = count + batch_count
        mean = mean + delta * (batch_count / total)
        m2 = m2 + batch_m2 + np.outer(delta, delta) * (count * batch_count / total)
        return mean, m2, total

    def _compute_valid_bands_mask(self, no_data: float | int, block_size: int) -> np.ndarray:
        valid_bands_mask = np.zeros(self.bands, dtype=bool)
        for _, _, block_data in self._iter_blocks(block_size):
            if np.issubdtype(block_data.dtype, np.floating):
                invalid = np.isnan(block_data) | (block_data == no_data)
            else:
                invalid = block_data == no_data
            valid_bands_mask |= np.any(~invalid, axis=(1, 2))
            if np.all(valid_bands_mask):
                break
        return valid_bands_mask

    def _compute_mean_cov(self, band_indices: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
        mean = None
        m2 = None
        count = 0
        for i, j, block_data in self._iter_blocks(block_size):
            block_data = block_data[band_indices, :, :]
            block_mask = self.backward_mask[i:i + block_data.shape[1], j:j + block_data.shape[2]]
            block_pixels = block_data[:, block_mask]
            if block_pixels.size == 0:
                continue
            block_pixels = block_pixels.T
            mean, m2, count = self._update_mean_cov(mean, m2, count, block_pixels)
        if count < 2:
            raise ValueError("Not enough valid pixels to compute statistics.")
        return mean, m2 / (count - 1)

    def _compute_noise_cov(self,
                           band_indices: np.ndarray,
                           block_size: int,
                           row_slice: slice | None,
                           col_slice: slice | None,
                           use_mask: bool) -> np.ndarray:
        row_slice = slice(None) if row_slice is None else row_slice
        col_slice = slice(None) if col_slice is None else col_slice
        row_start, row_stop, row_step = row_slice.indices(self.rows)
        col_start, col_stop, col_step = col_slice.indices(self.cols)
        if row_step != 1 or col_step != 1:
            raise ValueError("row_slice/col_slice step must be 1")
        row_end = min(row_stop, self.rows - 1)
        col_end = min(col_stop, self.cols - 1)
        mean = None
        m2 = None
        count = 0
        for i in range(row_start, row_end, block_size):
            for j in range(col_start, col_end, block_size):
                rows_valid = min(block_size, row_end - i)
                cols_valid = min(block_size, col_end - j)
                rows_read = rows_valid + 1
                cols_read = cols_valid + 1
                block_data = self.dataset.ReadAsArray(xoff=j, yoff=i, xsize=cols_read, ysize=rows_read)
                if block_data.ndim == 2:
                    block_data = block_data[np.newaxis, :, :]
                block_data = block_data[band_indices, :, :]
                deltas = block_data[:, :rows_valid, :cols_valid] - block_data[:, 1:rows_valid + 1, 1:cols_valid + 1]
                if use_mask:
                    block_mask = self.backward_mask[i:i + rows_read, j:j + cols_read]
                    valid_pairs = block_mask[:rows_valid, :cols_valid] & block_mask[1:rows_valid + 1, 1:cols_valid + 1]
                else:
                    valid_pairs = np.ones((rows_valid, cols_valid), dtype=bool)
                if np.issubdtype(deltas.dtype, np.floating):
                    valid_pairs &= np.all(np.isfinite(deltas), axis=0)
                deltas = deltas[:, valid_pairs]
                if deltas.size == 0:
                    continue
                deltas = deltas.T
                mean, m2, count = self._update_mean_cov(mean, m2, count, deltas)
        if count < 2:
            raise ValueError("Not enough samples to estimate noise statistics.")
        return (m2 / (count - 1)) / 2.0

    def _apply_linear_transform(self,
                                transform: np.ndarray,
                                mean: np.ndarray,
                                band_indices: np.ndarray,
                                block_size: int,
                                no_data: float | int) -> None:
        self.enhance_data = np.full((self.rows, self.cols, transform.shape[1]), no_data, dtype=np.float32)
        for i, j, block_data in self._iter_blocks(block_size):
            block_data = block_data[band_indices, :, :]
            block_mask = self.backward_mask[i:i + block_data.shape[1], j:j + block_data.shape[2]]
            block_pixels = block_data[:, block_mask]
            if block_pixels.size == 0:
                continue
            block_pixels = block_pixels.T.astype(np.float64, copy=False)
            block_pixels -= mean
            transformed = block_pixels @ transform
            block_out = self.enhance_data[i:i + block_data.shape[1], j:j + block_data.shape[2]]
            block_out[block_mask] = transformed.astype(np.float32, copy=False)
    
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
