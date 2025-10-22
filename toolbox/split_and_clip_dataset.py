import sys
sys.path.append('.')
from gdal_utils import batch_random_split_shp, clip_by_multishp
import os
from osgeo import gdal

input_tif = r'd:\OneDrive - cugb.edu.cn\项目数据\张川铀资源\张川项目\演示数据\research_area1_3bands.dat' # 裁剪区域栅格影像
input_shp_dir = r'd:\OneDrive - cugb.edu.cn\项目数据\张川铀资源\张川项目\演示数据\shp样本'
output_dir = r'c:\Users\85002\Desktop\test'
num_to_select = 0.6 # 要随机选取的要素数量, 如果小于1将按照比例选取
block_size = 17 # 样本块大小

if __name__ == '__main__':
    dataset = gdal.Open(input_tif)
    if dataset is None:
        raise ValueError(f"无法打开文件: {input_tif}")
    geo_transform = dataset.GetGeoTransform()
    pixel_size = geo_transform[1]
    batch_random_split_shp(input_shp_dir, output_dir, num_to_select, pixel_size=pixel_size)
    dataset = None
    print('Shapefile split completed!')

    train_dir = os.path.join(output_dir, 'train_dataset')
    test_dir = os.path.join(output_dir, 'test_dataset')
    clip_by_multishp(train_dir, input_tif, os.path.join(output_dir, 'split_part1'), block_size=block_size)
    print('Train dataset clipping completed!')
    clip_by_multishp(test_dir, input_tif, os.path.join(output_dir, 'split_part2'), block_size=block_size)
    print('Test dataset clipping completed!')