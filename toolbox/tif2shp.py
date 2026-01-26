"""批量栅格转矢量"""
import sys, os
sys.path.append('.')
import numpy as np
from gdal_utils import batch_raster_to_vector

input_shp_dir = r''
output_shp_dir = r''
delete_value = -1 # 需要删除的值

if __name__ == '__main__':
    batch_raster_to_vector(input_shp_dir, output_shp_dir, delete_value=delete_value)