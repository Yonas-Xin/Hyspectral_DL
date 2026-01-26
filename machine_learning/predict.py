import sys, os
sys.path.append('.')
from core import Hyperspectral_Image
import numpy as np
import pickle
from utils import label_to_rgb
import matplotlib.pyplot as plt

input_data_path = r''
input_model_path = r''
out_tif_path = r'rf_result.tif'
if __name__ == '__main__':
    print("Processing data...")
    img = Hyperspectral_Image()
    img.init(input_data_path)
    with open(input_model_path, 'rb') as f:
        clf_loaded = pickle.load(f)

    predict_whole_map = np.empty((img.rows,img.cols), dtype=np.uint8) + 255 # 背景值为-1
    for image_block, background_mask, i, j in img.image_block_iter(block_size=512, patch_size=1):
        print(f'Processing block at position ({i}, {j}) with shape {image_block.shape}...')
        rows, cols = background_mask.shape
        predict_map = np.zeros((rows, cols), dtype=np.uint8) + 255 # 初始化一个空的预测map，255代表背景值
        if np.any(background_mask == True): # 如果
            idx = 0
            dataset = image_block.transpose(1,2,0)[background_mask] # 只对非背景像素进行预测
            predict_data = np.empty(len(dataset), dtype=np.uint8) # 预分配内存，用来储存预测结果
            predict_map[background_mask] = clf_loaded.predict(dataset)
        predict_whole_map[i:i+rows, j:j+cols] = predict_map # 将预测结果填入整体预测矩阵
        print(f'Block at position ({i}, {j}) processed!')
    img.save_tif(out_tif_path, predict_whole_map, nodata=255) # 保存为tif文件
    map = label_to_rgb(predict_whole_map, background_value=255)
    plt.imsave(out_tif_path[:-4]+'.png', map, dpi=300)
    print("The result has been saved:", out_tif_path)