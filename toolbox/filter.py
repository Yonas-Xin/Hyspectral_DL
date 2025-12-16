import sys, os
sys.path.append('.')
from core import Hyperspectral_Image

input_file = r'C:\Users\85002\OneDrive - cugb.edu.cn\Word\小组项目\240922张川-铀矿探测\2025\年度报告\年度报告测试\矢量化\predict1_202512102056.tif' 
output_file = r'C:\Users\85002\OneDrive - cugb.edu.cn\Word\小组项目\240922张川-铀矿探测\2025\年度报告\年度报告测试\矢量化\predict1_opt2.tif'
remove_size = 64
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_file)
    img.sieve_filtering(output_file, threshold_pixels=remove_size)