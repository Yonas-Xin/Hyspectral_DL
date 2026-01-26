'''
根据样本.shp文件进行样本的裁剪
'''
import sys, os
sys.path.append('.')
from gdal_utils import clip_by_multishp
input_img = r'' # 裁剪区域栅格影像
input_dir = r"" # 裁剪shp文件夹
out_dir = r'' # 存储目录
block_size = 17

out_tif_name = "img"
if __name__ == "__main__":
    clip_by_multishp(out_dir, input_img, input_dir, block_size=block_size, out_tif_name=out_tif_name)