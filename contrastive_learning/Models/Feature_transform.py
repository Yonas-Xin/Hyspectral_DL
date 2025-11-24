import torch
import torch.nn as nn
import kornia.augmentation as K
from typing import Tuple
import random

from typing import Any, Dict, Optional, Tuple, Union
from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor, where
from kornia.geometry.bbox import bbox_generator, bbox_to_mask

class RandomErasing_Fixed(IntensityAugmentationBase2D):
    """随机擦除改写，擦除矩形区域的所有像元，但是必须保留图像块中心像元"""
    def __init__(
        self,
        scale: Union[Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self._param_generator = rg.RectangleEraseGenerator(scale, ratio, value)
        self.center_index = None

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        batch, c, h, w = input.size()
        if self.center_index == None:
            left_top = int(h / 2 - 1) if h % 2 == 0 else int(h // 2) # 计算中心位置左上坐标
            self.center_index = torch.ones((c,h,w))
            self.center_index[:, left_top, left_top] = 0
        values = params["values"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, *input.shape[1:]).to(input)

        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
        center_mask = self.center_index.unsqueeze(0).repeat(batch, 1, 1, 1).to(input)
        mask = mask * center_mask
        transformed = where(mask == 1.0, values, input)
        return transformed


class RandomSpectralMask(nn.Module):
    """随机掩膜每个像元光谱波段"""
    def __init__(self, mask_prob: float = 0.5, p: float = 0.5):
        super().__init__()
        self.mask_prob = mask_prob
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_prob == 0:
            return x
            
        # 为每个空间位置(H,W)生成独立的随机掩膜
        B, C, H, W = x.shape
        mask = torch.rand(B, C, H, W, device=x.device) > self.p
        batch_mask = torch.rand(B, 1, 1, 1, device=x.device) < self.mask_prob
        batch_mask = batch_mask.expand(B, C, H, W)
        mask = (mask | ~batch_mask).float()
        return x * mask

class BandDropout(nn.Module):
    """随机对某个光谱波段全丢弃"""
    def __init__(self, drop_prob=0.5, p=0.2):
        super().__init__()
        # self.dropout = nn.Dropout3d(p)  # 3D Dropout
        self.drop_prob = drop_prob
        self.p = p

    def forward(self, x):
        """
        输入: (batch, bands, H, W)
        输出: (batch, bands, H, W)，部分波段被随机置零
        """
        if not self.training:
            return x
        batch_size, C = x.shape[0], x.shape[1]
        drop_mask = (torch.rand(batch_size, 1, 1, 1, device=x.device) < self.drop_prob)
        band_mask = (torch.rand(batch_size, C, 1, 1, device=x.device) > self.p).float()
        mask = (drop_mask * band_mask + ~drop_mask)
        return x * mask

class HighDimBatchAugment(nn.Module):
    """高维图像块（如高光谱[B,C,H,W]）的批量增强"""
    def __init__(
            self,
            # crop_size: Tuple[int, int],
            spectral_mask_prob: float,
            band_dropout_prob:float,
            flip_prob: float = 0.5,
            rotate_prob: float = 0.5,
            add_gaussian_prob: float=0.5,
            erase_prob: float = 0.5,
            rotate_degrees: float = 90.0,
            # crop_scale: Tuple[float, float] = (0.8, 1.0),
            # crop_ratio: Tuple[float, float] = (0.9, 1.1),
            noise_std: float = 0.01,
            erase_scale: Tuple[float, float] = (0.01, 0.3),
            erase_ratio: Tuple[float, float] = (0.4, 2.5),
            spectral_mask_p: float = 0.25,
            bands_dropout_p: float = 0.25,
    ):
        super().__init__()
        # 初始化增强操作
        self.flip = K.RandomHorizontalFlip(p=flip_prob)
        self.rotate = K.RandomRotation(degrees=rotate_degrees, p=rotate_prob)
        # self.crop = K.RandomResizedCrop(
        #     size=crop_size,
        #     scale=crop_scale,
        #     ratio=crop_ratio,
        #     resample='bilinear'
        # )
        self.add_gaussian = K.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=add_gaussian_prob, same_on_batch=False
        )

        self.erase = RandomErasing_Fixed(
            p=erase_prob, scale=erase_scale, ratio=erase_ratio, value=0
        )
        if spectral_mask_prob > 0:
            self.spectral_mask = RandomSpectralMask(mask_prob=spectral_mask_prob, p=spectral_mask_p)
        else: self.spectral_mask = None
        if band_dropout_prob > 0:
            self.band_dropout = BandDropout(drop_prob=band_dropout_prob, p = bands_dropout_p)
        else: self.band_dropout = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, C, H, W]
        输出: [B, C, crop_H, crop_W]
        """
        # 确保输入是4D张量
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        elif x.dim() == 5:
            x = x.squeeze(1)
        else: pass
        # （所有操作自动支持批量）
        x = self.flip(x, inplace=True)  # 随机水平翻转
        x = self.rotate(x)  # 随机旋转
        # x = self.crop(x)  # 随机裁剪
        x = self.add_gaussian(x) # 随机添加高斯噪声
        x = self.erase(x) # 随机擦除
        if self.spectral_mask is not None:
            x = self.spectral_mask(x) # 光谱随机掩膜
        if self.band_dropout is not None:
            x = self.band_dropout(x) # 随机丢弃波段
        return x