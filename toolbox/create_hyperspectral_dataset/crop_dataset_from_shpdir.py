import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gdal_utils import clip_by_multishp
input_img = r'' # 裁剪区域栅格影像
input_dir = r"" # 裁剪shp文件夹
out_dir = r'' # 存储目录
block_size = 9

out_tif_name = "img"
output_format = "bin"
if __name__ == "__main__":
    clip_by_multishp(out_dir, input_img, input_dir, block_size=block_size, 
                     out_tif_name=out_tif_name, output_format=output_format)