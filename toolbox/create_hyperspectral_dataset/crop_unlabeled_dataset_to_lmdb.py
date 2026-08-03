import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import lmdb
import pickle
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from gdal_utils import write_data_to_tif, GDAL2NP_TYPE
import shutil
from core import Hyperspectral_Image


def get_row_valid_ranges(mask, patch_size):
    """
    生成基于行的有效像元起止坐标列表 (保留你原有的逻辑)
    """
    h, w = mask.shape
    coordinates_list = []
    # 找到所有包含 True 的行索引
    valid_rows_indices = np.where(np.any(mask, axis=1))[0]
    
    if len(valid_rows_indices) == 0:
        return []
    
    y_start = valid_rows_indices[0]
    for y in range(y_start, h, patch_size):
        if (y + patch_size//2) > h:
            break
        row_data = mask[y, :]
        valid_x_indices = np.where(row_data)[0]
        if len(valid_x_indices) > 0:
            x_first = valid_x_indices[0]
            x_last = valid_x_indices[-1]
            coordinates_list.append([(y, x_first), (y, x_last)])
        else:
            continue
    return coordinates_list

def write_images_to_lmdb(lmdb_path, img_path_list, mask_list, patch_size=30, 
                         commit_frequency=1000, map_size=1099511627776):
    """
    Args:
        lmdb_path (str): LMDB 数据库文件夹路径
        img_path_list (list): 影像文件路径列表
        mask_list (list): 对应的掩膜(Numpy Array)列表
        patch_size (int): 裁剪大小
        commit_frequency (int): 多少个 patch 提交一次事务
        map_size (int): LMDB 映射大小 (默认 1TB)
    """
    
    # 检查是否存在 LMDB，获取起始索引
    start_idx = 0
    if os.path.exists(lmdb_path):
        # 尝试读取现有的长度
        try:
            env_check = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env_check.begin() as txn:
                len_bytes = txn.get(b'__len__')
                if len_bytes:
                    start_idx = int(len_bytes.decode('ascii'))
            env_check.close()
            print(f"检测到已有数据库，将从索引 {start_idx} 开始追加数据。")
        except Exception:
            print("读取旧数据库失败，将从 0 开始。")
            start_idx = 0
    else:
        print("创建新数据库。")

    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    
    current_idx = start_idx
    batch_count = 0 # 用于控制 commit 频率

    # 辅助参数
    left_top = patch_size // 2 - 1 if patch_size % 2 == 0 else patch_size // 2
    right_bottom = patch_size // 2

    # 遍历每一张影像
    for img_idx, (sr_img_path, mask) in enumerate(zip(img_path_list, mask_list)):
        print(f"正在处理第 {img_idx + 1}/{len(img_path_list)} 张影像: {os.path.basename(sr_img_path)}")
        ds = gdal.Open(sr_img_path)
        if ds is None:
            print(f"无法打开 {sr_img_path}，跳过。")
            continue

        im_width = ds.RasterXSize
        im_height = ds.RasterYSize
        im_bands = ds.RasterCount
        im_geotrans = ds.GetGeoTransform()
        im_proj = ds.GetProjection()
        band = ds.GetRasterBand(1)
        dtype_name, numpy_dtype = GDAL2NP_TYPE.get(band.DataType, ('unknown', None)) # 确定numpy数据类型
        
        if numpy_dtype is None:
            print(f"不支持的数据类型，跳过 {sr_img_path}")
            continue

        # 获取有效行范围
        idx_list = get_row_valid_ranges(mask, patch_size)
        if not idx_list:
            continue

        # === 第一次遍历 (Center Based / 错位裁剪) ===
        for i in tqdm(idx_list[1:], desc=f"Img {img_idx+1} Loop 1", leave=False):
            start, end = i
            y, x_start_ = start
            y, x_end_ = end
            
            for x in range(x_start_ + patch_size, x_end_ + 1, patch_size):
                if mask[y, x] == False: continue
                if (x + patch_size//2) > x_end_: break 
                
                # 计算坐标
                x_start = x - left_top
                y_start = y - left_top
                x_end = x + right_bottom + 1
                y_end = y + right_bottom + 1

                # 读取逻辑
                read_x = max(0, x_start)
                read_y = max(0, y_start)
                read_width = min(x_end, im_width) - read_x
                read_height = min(y_end, im_height) - read_y
                
                if read_width > 0 and read_height > 0:
                    if im_bands > 1:
                        full_data = np.zeros((im_bands, patch_size, patch_size), dtype=numpy_dtype)
                        data = ds.ReadAsArray(read_x, read_y, read_width, read_height)
                        offset_x = read_x - x_start
                        offset_y = read_y - y_start
                        full_data[:, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
                    else:
                        full_data = np.zeros((patch_size, patch_size), dtype=numpy_dtype)
                        data = ds.GetRasterBand(1).ReadAsArray(read_x, read_y, read_width, read_height)
                        offset_x = read_x - x_start
                        offset_y = read_y - y_start
                        full_data[offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
                    # new_geotrans = list(im_geotrans) # 检验裁剪数据是否正确
                    # new_geotrans[0] = im_geotrans[0] + x_start * im_geotrans[1]
                    # new_geotrans[3] = im_geotrans[3] + y_start * im_geotrans[5]
                    # out_path = os.path.join(r'c:\Users\85002\Desktop\TempDIR\111', f"img_{current_idx}.tif")
                    # write_data_to_tif(out_path, full_data, new_geotrans, im_proj)

                    # --- 写入 LMDB ---
                    # Key 格式统一为: img_{索引}
                    key_bytes = f"img_{current_idx}".encode('ascii')
                    txn.put(key_bytes, pickle.dumps(full_data))
                    
                    current_idx += 1
                    batch_count += 1
                    
                    if batch_count % commit_frequency == 0:
                        txn.commit()
                        txn = env.begin(write=True)

        # === 第二次遍历 (Top-Left Based / 网格裁剪) ===
        for i in tqdm(idx_list, desc=f"Img {img_idx+1} Loop 2", leave=False):
            start, end = i
            y, x_start_ = start
            y, x_end_ = end
            if y + patch_size > im_height: break
            
            for x in range(x_start_, x_end_ + 1, patch_size): 
                if y+left_top >= im_height or x+left_top >= im_width: continue
                elif mask[y+left_top, x+left_top] == False: continue
                if x + patch_size > x_end_: break 
                
                x_start = x
                y_start = y
                x_end = x + patch_size
                y_end = y + patch_size

                read_x = max(0, x_start)
                read_y = max(0, y_start)
                read_width = min(x_end, im_width) - read_x
                read_height = min(y_end, im_height) - read_y
                
                if read_width > 0 and read_height > 0:
                    if im_bands > 1:
                        full_data = np.zeros((im_bands, patch_size, patch_size), dtype=numpy_dtype)
                        data = ds.ReadAsArray(read_x, read_y, read_width, read_height)
                        offset_x = read_x - x_start
                        offset_y = read_y - y_start
                        full_data[:, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
                    else:
                        full_data = np.zeros((patch_size, patch_size), dtype=numpy_dtype)
                        data = ds.GetRasterBand(1).ReadAsArray(read_x, read_y, read_width, read_height)
                        offset_x = read_x - x_start
                        offset_y = read_y - y_start
                        full_data[offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
                    # new_geotrans = list(im_geotrans)
                    # new_geotrans[0] = im_geotrans[0] + x_start * im_geotrans[1]
                    # new_geotrans[3] = im_geotrans[3] + y_start * im_geotrans[5]
                    # out_path = os.path.join(r'c:\Users\85002\Desktop\TempDIR\111', f"img_{current_idx}_middle.tif")
                    # write_data_to_tif(out_path, full_data, new_geotrans, im_proj)
                    # --- 写入 LMDB ---
                    key_bytes = f"img_{current_idx}".encode('ascii')
                    txn.put(key_bytes, pickle.dumps(full_data))
                    
                    current_idx += 1
                    batch_count += 1
                    
                    if batch_count % commit_frequency == 0:
                        txn.commit()
                        txn = env.begin(write=True)
        del ds # 关闭当前影像，释放内存

    # 结束处理，写入总长度
    print(f"写入元数据 __len__: {current_idx}")
    txn.put(b'__len__', str(current_idx).encode('ascii')) # 写入总长度
    txn.commit()
    env.close()
    print(f"全部完成。LMDB 中当前总数据量: {current_idx}")


def compact_lmdb(src_path, dst_path):
    """
    将源 LMDB 数据库压缩复制到新路径，如果成功则删除源数据库。
    """
    print(f"正在压缩: {src_path} -> {dst_path}")
    
    if not os.path.exists(src_path):
        print(f"错误：源路径 {src_path} 不存在！")
        return

    # 确保目标文件夹存在
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    else:
        if len(os.listdir(dst_path)) > 0:
            # 为了安全，如果目标已存在，先不操作
            print(f"目标路径已存在，且存在文件，跳过压缩: {dst_path}")
            return
        else:
            print(f"目标路径已存在但为空，继续操作: {dst_path}")
    # 执行压缩
    try:
        env = lmdb.open(src_path, readonly=True, lock=False)
        # compact=True 会丢弃未使用的空间
        env.copy(dst_path, compact=True)
        env.close()

        dst_data_file = os.path.join(dst_path, 'data.mdb')
        if not os.path.exists(dst_data_file) or os.path.getsize(dst_data_file) == 0:
            print("错误：压缩似乎失败了（目标文件为空），未删除源文件。")
            return

        # 对比大小
        src_size = os.path.getsize(os.path.join(src_path, 'data.mdb')) / (1024**3)
        dst_size = os.path.getsize(dst_data_file) / (1024**3)
        print(f"压缩成功！体积变化: {src_size:.2f} GB -> {dst_size:.2f} GB")

        # 删除源数据库
        print(f"正在删除源数据库: {src_path} ...")
        shutil.rmtree(src_path)
        print("源数据库已删除。")

    except Exception as e:
        print(f"发生错误，操作中止，源文件未删除。错误信息: {e}")


datapaths = [r''] 
patch_size = 13
target_size = 10 * 1024 * 1024 * 1024 # 10 GB
output = r''
compact_output = os.path.join(os.path.dirname(output), os.path.basename(output) + '_compact')
if __name__ == "__main__":
    masks = [Hyperspectral_Image(path, True).backward_mask for path in datapaths]
    write_images_to_lmdb(output, datapaths, masks, patch_size=patch_size, commit_frequency=1000
                         , map_size=target_size)
    compact_lmdb(output, compact_output)