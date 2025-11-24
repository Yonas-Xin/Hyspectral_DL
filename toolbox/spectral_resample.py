import sys
sys.path.append('.')
from algorithms import HyperspectralResampler
import numpy as np

input = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\haide\haide.dat'
output = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\haide\haide_resampled.tif'

satellite_name = 'ZY102E' # 输入高光谱的卫星传感器名称，在assts/resample_params文件夹中预定义
target_satellite_name = None # 目标重采样的卫星传感器名称，在assts/resample_params文件夹中预定义
target_center = np.arange(400, 2551, 10)  # 目标中心波段，400nm到2500nm，每10nm一个波段
target_fwhm = np.full_like(target_center, 10)  # 每个波段的FWHM为10nm, 为None则自动计算
delete_wavelengths = [(1350, 1450), (1800, 1950), (2450, 2550)]  # 重采样后删除指定波段范围列表
# 建议删除的波段范围: 1350-1450nm (水汽吸收带), 1800-1950nm (水汽吸收带)，2450nm-（边缘/水汽波段）
if __name__ == '__main__':
    resampler = HyperspectralResampler(input, satellite_name=satellite_name, 
                                       delete_wavelengths=delete_wavelengths)
    resampler.resample(output, target_satellite_name, target_center, target_fwhm) # target_satellite_name or target_center 必须提供其中一个