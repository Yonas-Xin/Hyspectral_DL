import torch
from cnn_model.Models.Encoder import ENCODER_REGISTRY
from contrastive_learning.Models.Decoder import *

class Contrastive_Model(nn.Module):
    def __init__(self, encoder_model_name, in_shape=None):
        """
        encoder_model_name: str encoder模型的名称
        """
        super().__init__()
        items = ENCODER_REGISTRY[encoder_model_name] # 获取注册表中的模型构造信息
        encoder, dim, in_channels, feature_maps = items
        self.encoder = encoder(in_shape=in_shape)
        self.decoder = Contrastive_Decoder(in_channels, 1024, mid_channels=1024)
        self.dim = dim
        self.embedding_dim = in_channels
        self.if_draw_feature_maps = feature_maps
    def forward(self, x):
        # 输入数据有以下两种形式: [B, bands], [B, C, H, W], [B, 1, C, H, W]
        if x.dim() == 2:
            if self.dim == 3: x = x.unsqueeze(1)
            else: raise ValueError(f'The input data dimensions {x.dim()} do not match the model requirements {self.dim}.')
        elif x.dim() == 4:
            if self.dim == 3:
                _, _, h, w = x.shape
                left_top = h // 2 - 1 if h % 2 == 0 else h // 2
                x = x[:, :, left_top, left_top]
                x = x.unsqueeze(1)
            elif self.dim == 4:
                pass
            elif self.dim == 5:
                x = x.unsqueeze(1)
            else: raise ValueError(f'The input data dimensions {x.dim()} do not match the model requirements {self.dim}.')
        elif x.dim() == 5:
            if self.dim == 3:
                x = x.squeeze(1)
                _, _, h, w = x.shape
                left_top = h // 2 - 1 if h % 2 == 0 else h // 2
                x = x[:, :, left_top, left_top]
                x = x.unsqueeze(1)
            elif self.dim == 4:
                x = x.squeeze(1)
            elif self.dim == 5:
                pass
            else: raise ValueError(f'The input data dimensions {x.dim()} do not match the model requirements {self.dim}.')
        else: raise ValueError("The input data dimensions must be [B, bands] or [B, C, H, W] or [B, 1, C, H, W]!")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__=='__main__':
    x = torch.randn(2,1,25,25,138)