import torch
from cnn_model.Models.Encoder import SRACN_Encoder, Res_3D_18Net_encoder, Res_3D_50Net_encoder, Res_3D_34Net_encoder, ResNet18_encoder, \
                                    ResNet34_encoder, ResNet50_encoder, SSRN_encoder, Vgg16_encoder, HybridSN_encoder, Common_1DCNN_Encoder, \
                                    Common_2DCNN_Encoder, Common_3DCNN_Encoder, MobileNetV1_encoder, MobileNetV2_encoder, SpecTransformer_encoder
from contrastive_learning.Models.Decoder import *

ENCODER_DICT = {
    'SRACN':SRACN_Encoder,
    'Common_1DCNN': Common_1DCNN_Encoder,
    'Common_2DCNN': Common_2DCNN_Encoder,
    'Common_3DCNN': Common_3DCNN_Encoder,
    "Res_3D_18Net": Res_3D_18Net_encoder,
    "Res_3D_34Net": Res_3D_34Net_encoder,
    "Res_3D_50Net": Res_3D_50Net_encoder,
    'SSRN': SSRN_encoder,
    'HybridSN': HybridSN_encoder,
    'Vgg16': Vgg16_encoder,
    'MobileNetV1': MobileNetV1_encoder,
    'MobileNetV2': MobileNetV2_encoder,
    'ResNet18': ResNet18_encoder,
    'ResNet34': ResNet34_encoder,
    'ResNet50': ResNet50_encoder,
    'spec_transformer': SpecTransformer_encoder,
}

DIM_DICT = {
    'SRACN':5,
    'Common_1DCNN': 3,
    'Common_2DCNN': 4,
    'Common_3DCNN': 5,
    "Res_3D_18Net": 5,
    "Res_3D_34Net": 5,
    "Res_3D_50Net": 5,
    'SSRN': 4,
    'HybridSN': 5,
    'Vgg16': 4,
    'MobileNetV1': 4,
    'MobileNetV2': 4,
    'ResNet18': 4,
    'ResNet34': 4,
    'ResNet50': 4,
    'spec_transformer': 4
}

IN_CHANNELS = {
    'SRACN':512,
    'Common_1DCNN': 512,
    'Common_2DCNN': 512,
    'Common_3DCNN': 512,
    "Res_3D_18Net": 512,
    "Res_3D_34Net": 512,
    "Res_3D_50Net": 512 * 4,
    'SSRN': 28,
    'HybridSN': 256,
    'Vgg16': 512,
    'MobileNetV1': 1024,
    'MobileNetV2': 1280,
    'ResNet18': 512,
    'ResNet34': 512,
    'ResNet50': 512 * 4,
    'spec_transformer': 768
}

FEATURE_MAPS = {
    'SRACN':True,
    'Common_1DCNN': False,
    'Common_2DCNN': True,
    'Common_3DCNN': True,
    "Res_3D_18Net": True,
    "Res_3D_34Net": True,
    "Res_3D_50Net": True,
    'SSRN': True,
    'HybridSN': True,
    'Vgg16': True,
    'MobileNetV1': True,
    'MobileNetV2': True,
    'ResNet18': True,
    'ResNet34': True,
    'ResNet50': True,
    'spec_transformer': False
}
class Contrastive_Model(nn.Module):
    def __init__(self, encoder_model_name, in_shape=None):
        """
        encoder_model_name: str encoder模型的名称
        """
        super().__init__()
        self.encoder = ENCODER_DICT[encoder_model_name](in_shape=in_shape)
        self.decoder = Contrastive_Decoder(IN_CHANNELS[encoder_model_name], 1024, mid_channels=1024)
        self.dim = DIM_DICT[encoder_model_name]
        self.embedding_dim = IN_CHANNELS[encoder_model_name]
        self.if_draw_feature_maps = FEATURE_MAPS[encoder_model_name] if hasattr(FEATURE_MAPS, encoder_model_name) else False
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