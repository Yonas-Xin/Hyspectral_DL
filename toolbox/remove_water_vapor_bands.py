import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core import Hyperspectral_Image
from algorithms import choose_satellite_params

input = r''
output = r''  # 输出影像路径

satellite_name = 'ZY1F'  # 卫星名称，当影像无波长信息时使用，在 assets/resample_params 中预定义
water_vapor_ranges = [(300, 400), (1340, 1450), (1800, 1950), (2450, 2600)]  # 水汽吸收波段范围 (nm)，可自定义
remove_overlap = True  # 是否去除 VNIR-SWIR 重合波段

"""
PS: ZY1E卫星的短波红外影像一般有坏波段[22, 23, 24, 25, 50, 51, 52, 53, 54, 55](1-based)
在整个波段索引中为[98, 99, 100, 101, 126, 127, 128, 129, 130, 131](1-based)
其坏波段的波长范围为[1360-1415nm, 1835-1920nm], 几乎在常用的水汽吸收波段范围内
"""
if __name__ == '__main__':
    img = Hyperspectral_Image(input=input, init_fig=False)

    # 如果影像本身没有波长信息，则从卫星参数资源文件中读取
    if img.wavelengths is None:
        print(f"INFO: 影像中无波长信息，回退使用卫星参数: {satellite_name}")
        wavelengths, fwhms = choose_satellite_params(satellite_name)
        img.wavelengths = wavelengths
        img.fwhm = fwhms

    img.subset_image_from_wavelength(
        output_path=output,
        water_vapor_ranges=water_vapor_ranges,
        remove_overlap=remove_overlap,
    )