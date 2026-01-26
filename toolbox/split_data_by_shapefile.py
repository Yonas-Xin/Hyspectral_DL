import sys
sys.path.append('.')
from gdal_utils import batch_random_split_shp

if __name__ == '__main__':
    input_shp_dir = r''
    output_shp_dir = r''
    num_to_select = 0.6 # 要随机选取的要素数量, 如果小于1将按照比例选取
    batch_random_split_shp(input_shp_dir, output_shp_dir, num_to_select)