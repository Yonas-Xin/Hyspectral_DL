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
batch = 256
input_data = r""
model_pth = r'.\model.pt'  # 模型路径
MUTITHREADING_MODE = False # 是否使用多线程加速数据加载, 实测Fasle时速度更快，根据情况使用
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

    # 设置绘制图像的参数
    out_png = f"{output_path[:-4]}.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    global_rgb_vis = utils.label_to_rgb(predict_whole_map, background_value=255) 
    ax1.imshow(img.ori_img)
    ax1.axis('off')
    pred_imshow_obj = ax2.imshow(global_rgb_vis) # 获取 image object 引用，后续通过它更新数据
    ax2.axis('off')
    plt.tight_layout()

    model = torch.load(model_pth, weights_only=False, map_location=device)
    model.to(device) # 移动模型到指定设备
    model.eval() # 设置为评估模式
    num_workers = cpu_count() // 4 if MUTITHREADING_MODE else 0
    dataset = Predict_Dataset(patch_size)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=num_workers, 
                            pin_memory=True if num_workers > 0 else False) # 实测这里numworker变多不会加快速率，反而会因为加载线程拖慢速度
    with torch.no_grad():
        for image_block, background_mask, i, j in img.image_block_iter(block_size=image_block_size, patch_size=patch_size):
            rows, cols = background_mask.shape
            predict_map = np.zeros((rows, cols), dtype=np.uint8) + 255 # 初始化一个空的预测map，-1代表背景值
            if np.any(background_mask): # 如果
                idx = 0
                dataset.update_data(image_block, background_mask) # 更新dataset后调用dataloader会重新启动进程
                predict_data = torch.empty(len(dataset), dtype=torch.uint8, device=device) # 预分配内存，用来储存预测结果
                for data in tqdm(dataloader, total=len(dataloader), desc='Block Predicting'):
                    batch = data.shape[0]
                    data = data.to(device)
                    outputs = model(data)
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(0)
                    _, predicted = torch.max(outputs, 1)
                    predict_data[idx:idx + batch, ] = predicted
                    idx += batch
                predict_map[background_mask] = predict_data.cpu().numpy() if predict_data.device.type == 'cuda' else predict_data.numpy() # 将预测结果填入对应位置
                predict_whole_map[i:i+rows, j:j+cols] = predict_map # 将预测结果填入整体预测矩阵

                # 更新预测过程中的图像
                block_rgb = utils.label_to_rgb(predict_map, background_value=255)
                global_rgb_vis[i:i+rows, j:j+cols] = block_rgb
                pred_imshow_obj.set_data(global_rgb_vis)
                fig.savefig(out_png, dpi=150)
    img.save_tif(output_path, predict_whole_map, nodata=255) # 最终保存为tif文件