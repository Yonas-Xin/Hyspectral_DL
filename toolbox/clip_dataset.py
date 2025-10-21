'''
根据样本.shp文件进行样本的裁剪
'''
import sys, os
sys.path.append('.')
from gdal_utils import clip_by_multishp
input_img = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat' # 裁剪区域栅格影像
input_dir = r"c:\Users\85002\OneDrive\文档\小论文\dataset11classes\d6-4new\split_part1" # 裁剪shp文件夹
out_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset11classes\d6-4new\测试集1' # 存储目录
block_size = 17

out_tif_name = "img"
if __name__ == "__main__":
    clip_by_multishp(out_dir, input_img, input_dir, block_size=block_size, out_tif_name=out_tif_name)