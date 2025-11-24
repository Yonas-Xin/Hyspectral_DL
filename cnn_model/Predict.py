"""将大幅高光谱影像进行分块的滑窗预测，避免占用大量显存
预测结果是一个二维矩阵，255代表背景, 其余值代表预测的地物类别, 最多预测256类（包括背景）"""
import sys, os
sys.path.append('.')
from cnn_model.Models.Data import Predict_Dataset
from torch.utils.data import DataLoader
import utils
from core import Hyperspectral_Image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import cpu_count
import matplotlib
import re
matplotlib.use('Agg')

output_path = 'SRACN_PRE.tif'
batch = 128
input_data = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat"
model_pth = r'cnn_model\_results\Common_1DCNN_Test_Patch17_202511241428_ID9iuaw3s6bopxyubfbtdog\Common_1DCNN_Test_Patch17_202511241428_ID9iuaw3s6bopxyubfbtdog_best.pt'  # 模型路径
MUTITHREADING_MODE = False # 是否使用多线程加速数据加载, 实测Fasle时速度更快，根据情况使用
DRAW_RGB = True # 是否绘制预测过程中的rgb图像
rgb_combine = (25,15,5) #(29,19,9) # 绘制图像时的rgb组合，从1开始, 如果无效则使用第一个波段, 图像太大时一定程度上会影响速度
image_block_size = 512 # 分块预测时每个大块的大小，越大越占用内存，但预测速度越快
if __name__ == '__main__':
    patch_size = re.search(r'Patch(\d+)', os.path.basename(model_pth))
    if patch_size is None:
        raise ValueError("Patch size not found in model path! Please ensure the model path contains 'PatchXX' indicating the patch size.")
    patch_size = int(patch_size.group(1))
    print(f"🎯 The Patch Size: {patch_size}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    left_top = int(patch_size / 2 - 1) if patch_size % 2 == 0 else int(patch_size // 2)
    current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
    output_path = f"{output_path[:-4]}_{current_time}.tif"
    img = Hyperspectral_Image()
    img.init(input_data, rgb=rgb_combine)
    predict_whole_map = np.empty((img.rows,img.cols), dtype=np.uint8) + 255 # 背景值为-1
    model = torch.load(model_pth, weights_only=False, map_location=device)
    model.to(device)
    model.eval()
    if DRAW_RGB:
        out_png = f"{output_path[:-4]}.png"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img.ori_img)
        ax1.axis('off')
        ax2.axis('off')

    num_workers = cpu_count() // 4 if MUTITHREADING_MODE else 0
    dataset = Predict_Dataset(patch_size)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=num_workers, 
                            pin_memory=True if num_workers > 0 else False) # 实测这里numworker变多不会加快速率，反而会因为加载线程拖慢速度
    with torch.no_grad():
        for image_block, background_mask, i, j in img.image_block_iter(block_size=image_block_size, patch_size=patch_size):
            rows, cols = background_mask.shape
            predict_map = np.zeros((rows, cols), dtype=np.uint8) + 255 # 初始化一个空的预测map，-1代表背景值
            if np.any(background_mask == True): # 如果
                idx = 0
                dataset.update_data(image_block, background_mask) # 更新dataset后调用dataloader会重新启动进程
                predict_data = torch.empty(len(dataset), dtype=torch.uint8, device=device) # 预分配内存，用来储存预测结果
                for data in tqdm(dataloader, total=len(dataloader), desc=f'Block Predicting'):
                    batch = data.shape[0]
                    data = data.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    predict_data[idx:idx + batch, ] = predicted
                    idx += batch
                predict_map[background_mask] = predict_data.cpu().numpy() if predict_data.device.type == 'cuda' else predict_data.numpy() # 将预测结果填入对应位置
                predict_whole_map[i:i+rows, j:j+cols] = predict_map # 将预测结果填入整体预测矩阵
                img.save_tif(output_path, predict_whole_map, nodata=255) # 保存为tif文件

                # 下面保存预测过程中的图像
                if DRAW_RGB:
                    map = utils.label_to_rgb(predict_whole_map, background_value=255)
                    ax2.imshow(map)
                    fig.savefig(out_png, bbox_inches='tight', dpi=150)