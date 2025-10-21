from gdal_utils import batch_random_split_shp, clip_by_multishp
import os



input_tif = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat' # 裁剪区域栅格影像
input_shp_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset11classes\d6-4new\split_part1'
output_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset11classes\test'
num_to_select = 0.6 # 要随机选取的要素数量, 如果小于1将按照比例选取
block_size = 17 # 样本块大小

pixel_size = 29 # 影像的像素大小

if __name__ == '__main__':
    batch_random_split_shp(input_shp_dir, output_dir, num_to_select)
    print('Shapefile split completed!')

    train_dir = os.path.join(output_dir, 'train_dataset')
    test_dir = os.path.join(output_dir, 'test_dataset')
    clip_by_multishp(train_dir, input_tif, os.path.join(output_dir, 'split_part1'), block_size=block_size)
    print('Train dataset clipping completed!')
    clip_by_multishp(test_dir, input_tif, os.path.join(output_dir, 'split_part2'), block_size=block_size)
    print('Test dataset clipping completed!')