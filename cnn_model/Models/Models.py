"""针对Models模块的升级"""
from cnn_model.Models.Encoder import *
from cnn_model.Models.Decoder import *
import torch.nn as nn
import torch.nn.functional as F
import torch

class My_Model(nn.Module):
    def __init__(self, out_classes=None, out_embedding=None, in_shape=None): # 框架需要三个输入
        super().__init__()
        self.lock_grad = True
        self.unfreeze_idx = None

    def _freeze_encoder(self):
        if self.lock_grad:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print('The Encoder parameters have been frozen.')
        else:
            print('Attention: The Encoder parameters are not frozen! Cause: Too many parameters do not match!')
        
        for param in self.encoder.fc.parameters():
            param.requires_grad = True # encoder 的fc层需要正常梯度传播
        

    def _load_encoer_params(self, state_dict):
        try:
            self.encoder.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # 过滤掉不匹配的键
            model_state_dict = self.encoder.state_dict()
            matched_state_dict = {
                k: v for k, v in state_dict.items() 
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            model_state_dict.update(matched_state_dict)
            self.encoder.load_state_dict(model_state_dict, strict=False)
            skipped = set(state_dict.keys()) - set(matched_state_dict.keys())
            if skipped:
                print(f"Atteneion: The encoder weights do not match exactly! Skipped loading these keys due to size mismatch: {skipped}")
            if len(skipped) <= 2 and 'fc.weight' in skipped: # 只允许fc层参数不匹配
                self.lock_grad = True
            else:
                self.lock_grad = False
                print(f"{len(skipped)} parameters are skipped, Please check if the pre-trained model is consistent with the parameters of the training model. ")

class Res_3D_18Net(My_Model):
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = Res_3D_18Net_encoder(out_embedding=out_embedding) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Res_3D_34Net(My_Model):
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = Res_3D_34Net_encoder(out_embedding=out_embedding) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Res_3D_50Net(My_Model):
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = Res_3D_50Net_encoder(out_embedding=out_embedding) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Common_3DCNN(My_Model):
    '''浅层3D CNN模型'''
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = Common_3DCNN_Encoder(out_embedding=out_embedding) # 3d卷积残差编码器
        self.decoder = nn.Linear(128, out_features=out_classes)
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Common_1DCNN(My_Model):
    '''浅层1D CNN模型'''
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = Common_1DCNN_Encoder(out_embedding=out_embedding)  # 1D CNN 编码器
        self.decoder = deep_classfier(128, out_classes, mid_channels=1024)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, bands]
        elif x.dim() == 4: # [B, c, h, w]
            _, _, h, w = x.shape
            left_top = h // 2 - 1 if h % 2 == 0 else h // 2
            x = x[:, :, left_top, left_top]
            x = x.unsqueeze(1)
        elif x.dim() == 5: # [B, 1, c, h, w]
            x = x.squeeze(1)
            _, _, h, w = x.shape
            left_top = h // 2 - 1 if h % 2 == 0 else h // 2
            x = x[:, :, left_top, left_top]
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected input dimension 2 or 3, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Common_2DCNN(My_Model):
    '''浅层1D CNN模型'''
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = Common_2DCNN_Encoder(out_embedding=out_embedding, in_shape=in_shape)  # 1D CNN 编码器
        self.decoder = deep_classfier(128, out_classes, mid_channels=1024)

    def forward(self, x):
        if x.dim() == 5: # [B, c, h, w]
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class SRACN(My_Model):
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        if in_shape is None:
            raise ValueError("in_shape must be provided for the model.")
        self.encoder = SRACN_Encoder(in_shape=in_shape, out_embedding=out_embedding)
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)

    def _freeze_encoder(self):
        super()._freeze_encoder()
        #    这里自由解冻其他层
        # for param in self.encoder.res_block6.parameters():
        #     param.requires_grad = True
        # print('最外层已解冻')

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ==============================其他论文中的模型==============================
class SSRN(My_Model):
    """code from: https://github.com/zilongzhong/SSRN SSRN以4D 为输入"""
    def __init__(self, out_classes, out_embedding=None, in_shape=None):
        super(SSRN, self).__init__()
        self.encoder = SSRN_encoder(in_shape=in_shape)
        self.decoder = nn.Linear(128, out_classes)

    def forward(self, x):
        if x.dim() == 5: # [B, c, h, w]
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class HybridSN(My_Model):
    """code from: https://github.com/gokriznastic/HybridSN
    自适应输入维度"""
    def __init__(self, out_classes, out_embedding=None, in_shape=None):
        super(HybridSN, self).__init__()
        self.encoder = HybridSN_encoder(in_shape = in_shape)
        self.decoder = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256,128),
            nn.Dropout(0.4),
            nn.Linear(128, out_classes)
        )
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Vgg16_net(My_Model):
    """code from: https://github.com/Lornatang/VGG-PyTorch
    为了适应小patch数据, 做了点pool的小修改"""
    def __init__(self, out_classes, out_embedding=None, in_shape=None):
        super().__init__()
        self.encoder=Vgg16_encoder(in_shape=in_shape)

        self.decoder=nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256,out_classes)
        )

    def forward(self,x):
        if x.dim() == 5: # [B, c, h, w]
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x=self.encoder(x)
        x=self.decoder(x)
        return x

class MobileNetV1(My_Model):
    """code from: https://developer.aliyun.com/article/1309561
    论文地址: https://arxiv.org/abs/1704.04861"""
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder=MobileNetV1_encoder(out_embedding=out_embedding, in_shape=in_shape)
        self.decoder = nn.Linear(out_embedding, out_classes)
    
    def forward(self,x):
        if x.dim() == 5: # [B, c, h, w]
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")        
        x=self.encoder(x)
        x=self.decoder(x)
        return x

class MobileNetV2(My_Model):
    """code from: https://blog.csdn.net/Code_and516/article/details/130200844
    论文地址: https://arxiv.org/abs/1801.04381"""
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder=MobileNetV2_encoder(out_embedding=out_embedding, in_shape=in_shape)
        self.decoder = nn.Linear(out_embedding, out_classes)
    
    def forward(self,x):
        if x.dim() == 5: # [B, c, h, w]z
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")    
        x=self.encoder(x)
        x=self.decoder(x)
        return x
    
class ResNet18(My_Model):
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = ResNet18_encoder(out_embedding=out_embedding, in_shape=in_shape)
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class ResNet34(My_Model):
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = ResNet34_encoder(out_embedding=out_embedding, in_shape=in_shape)
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class ResNet50(My_Model):
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = ResNet50_encoder(out_embedding=out_embedding, in_shape=in_shape)
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
MODEL_DICT = {
    'SRACN':SRACN,
    'Shallow_1DCNN':Common_1DCNN,
    'Shallow_3DCNN':Common_3DCNN,
    "Res_3D_18Net": Res_3D_18Net,
    "Res_3D_34Net": Res_3D_34Net,
    "Res_3D_50Net": Res_3D_50Net
}