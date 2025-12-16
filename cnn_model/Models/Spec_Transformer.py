import torch
import torch.nn as nn
from torch.nn import Dropout, Conv3d
from cnn_model.Models.Transformer import Transformer

def get_base_configs():
    configs = {
        'hidden_size': 768,
        'split_size': 3, # 每个patch在通道维度的大小
        'num_heads': 12,
        'num_layers': 12,
        'attention_dropout_rate': 0.0,
        'dropout_rate': 0.1,
        'learnable_pos_embed': False,
    }
    return configs

class PatchEmbeddings(nn.Module): # 将图像块划分为patches，并映射到隐藏空间, 同时返回原始patches，用于MAE重构
    """基于Linear的Patch Embedding，将图像划分为若干补丁，并映射到隐藏空间。"""
    def __init__(self, img_size, config):
        super(PatchEmbeddings, self).__init__()
        C, H, W = img_size
        self.split_size = config['split_size']  # 每个补丁在通道维度的大小
        C = self._make_divisible(C, self.split_size)
        self.n_patches = (C // self.split_size) # 从通道维度切分补丁
        self.patch_embed = nn.Linear(self.split_size * H * W, config['hidden_size'])

    def forward(self, x):
        patches = x.view(x.shape[0], self.n_patches, self.split_size, x.shape[-2], x.shape[-1])
        patches = patches.flatten(2) # (b, n_patches, split_size*H*W)
        x = self.patch_embed(patches) # (b, n_patches, hidden_size)
        return x, patches
    
    def _make_divisible(self, C, divisor): # 保证C能被divisor整除
        remainder = C % divisor
        if remainder != 0:
            print(f'Warning: Channels {C} is not divisible by divisor {divisor}, \
                  we will pad to {C + divisor - remainder} channels to make it divisible.')
            return C + divisor - remainder
        return C

class Embeddings(nn.Module):
    """
    基于Conv3d的Patch Embedding，将图像划分为若干补丁，并映射到隐藏空间。
    不适用与MAE，仅作为参考。
    """
    def __init__(self, img_size, config):
        super(Embeddings, self).__init__()
        C, H, W = img_size
        self.split_size = config['split_size']  # 每个补丁在通道维度的大小
        assert C % self.split_size == 0, '请确保数据的通道数必须能被split_size整除'
        n_patches = (C // self.split_size) # 从通道维度切分补丁

        self.patch_embeddings = Conv3d(in_channels=1,
                                       out_channels=config['hidden_size'],
                                       kernel_size=(self.split_size, H, W),  # 等,Patch 的大小，确保每个 Patch 的信息被映射到一个嵌入向量中
                                       stride=(self.split_size, H, W))  # 等于 Patch 的大小，保证不重叠分割
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # (b, c, w, h) -> (b, c, 1, w, h) 变成5维以适应Conv3d
        x = self.patch_embeddings(x)  # x.shape (b, hidden_size, w/patch_size, h/patch_size)
        x = x.flatten(2)  # x.shape (b, hidden_size, w/patch_size*h/patch_size) 下面用N表示w/patch_size*h/patch_size
        x = x.transpose(-1, -2)  # x.shape (b, N, hidden_size)
        return x

class SpecTransformer(nn.Module):
    def __init__(self, img_size, config=get_base_configs()):
        super(SpecTransformer, self).__init__()
        self.patch_emb = PatchEmbeddings(img_size, config)
        self.transformer = Transformer(config)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_emb.n_patches + 1, config['hidden_size'])
                                                , requires_grad=config['learnable_pos_embed'])# 绝对位置编码
        nn.init.trunc_normal_(self.position_embeddings, std=.02) # 初始化位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))
        nn.init.normal_(self.cls_token, std=.02)  # 初始化cls token

        self.emb_dropout = Dropout(config['attention_dropout_rate']) # 嵌入层的dropout
        self.hidden_size = config['hidden_size'] # 用于模型外部获取hidden_size

    def forward(self, x):
        x = self._make_channels_divisible(x, self.patch_emb.split_size) # 保证通道数是split_size的整数倍
        x, _ = self.patch_emb(x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)  # x.shape (b, N+1, hidden_size)
        x += self.position_embeddings  # 加上位置信息
        x = self.emb_dropout(x)
        x = self.transformer(x)
        return x[:, 0]  # 直接返回CLS token的编码
    
    def _make_channels_divisible(self, x, divisor): # 如果通道数不是divisor的整数倍，则进行padding
        if len(x.shape) == 4:
            _, C, _, _ = x.shape
        else:
            C, _, _ = x.shape
        remainder = C % divisor
        if remainder == 0:
            return x
        pad_channels = divisor - remainder
        return nn.ConstantPad2d((0, 0, 0, 0, 0, pad_channels), 0)(x)