"""Attention: 为了统一初始化参数, 所有模型都必须有in_shape作为初始化数据"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from classifier_model.Models.Spec_Transformer import SpecTransformer, get_base_configs
from classifier_model.Models.Decoder import Contrastive_Decoder
ENCODER_REGISTRY = {}

def register_encoder(name: str | None = None, 
                     dim: int = 5, # 编码器接收输入数据的维度
                     out_channels: int = 128, # 编码器输出的特征维度
                     feature_map: bool = False): # 编码器是否能够输出特征图
    """模型注册装饰器, name缺省则用类/函数的 __name__"""
    def wrapper(obj):
        key = name or obj.__name__
        if key in ENCODER_REGISTRY:
            raise ValueError(f"模型名称重复: {key}")
        ENCODER_REGISTRY[key] = [obj, dim, out_channels, feature_map]
        return obj
    return wrapper

@register_encoder(name='SRACN', dim=5, out_channels=128, feature_map=True)
class SRACN_Encoder(nn.Module):
    '''6个残差块和一个卷积块'''
    def __init__(self, in_shape=None):
        super().__init__()
        bands, H, W = in_shape
        self.spectral_attention = ECA_SpectralAttention_3d(bands)
        self.conv_block = Common_3d(1, 64, 7, stride=(2,1,1), padding=(3))
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.res_block1 = Residual_block(64, 64, (3,3,3), (1,1,1), 1)
        self.res_block2 = Residual_block(64, 128, (3,3,3), (1,1,1), (2,1,1)) # stride=2
        self.res_block3 = Residual_block(128, 128, (3,3,3), (1,1,1), 1)
        self.res_block4 = Residual_block(128, 256, (3,3,3), (1,1,1), (2,1,1)) # stride=2
        self.res_block5 = Residual_block(256, 256, (3,3,3), (1,1,1), 1)
        self.res_block6 = Residual_block(256, 512, (3,3,3), (1,1,1), (2,1,1)) # stride=2
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1)) # 立方体压缩
        self.fc = nn.Linear(512, 128) # 输出特征维度为 128
    def forward(self, x):
        x = self.spectral_attention(x)
        x = self.pool(self.conv_block(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =================================================================================================
# 编码器组件
# =================================================================================================
class Common_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1), stride=1):
        super(Common_3d,self).__init__()
        '''先batch，后激活'''
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))

class Residual_block(nn.Module):
    '''标准残差块结构'''
    def __init__(self, in_channel=1, out_channel=64,kernel_size=(3,3,3), padding=(1,1,1), stride=1):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm3d(out_channel),
            )
        if in_channel!=out_channel:
            self.use_downsample = True
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1), stride=stride, bias=False),
                nn.BatchNorm3d(out_channel)
            )
        else:self.use_downsample = False
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.bottleneck(x)
        if self.use_downsample:
            x = self.downsample(x)
        return self.relu(out+x)
 
class Basic_Residual_block(nn.Module):
    """基础残差块"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic_Residual_block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample   #对输入特征图大小进行减半处理
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        return out

class Bottleneck_Residual_block(nn.Module):
    """瓶颈残差块"""
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Residual_block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        return out

class Basic_Residual_block_2d(nn.Module):
    """基础残差块"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic_Residual_block_2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample   #对输入特征图大小进行减半处理
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        return out

class Bottleneck_Residual_block_2d(nn.Module):
    """瓶颈残差块"""
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Residual_block_2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        return out

# ============ECA 光谱注意力组件============
class ECA_SpectralAttention_3d(nn.Module):
    def __init__(self, bands,gamma=2,b=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d((bands,1, 1))  # 压缩空间维度 (rows,cols) → (1,1)
        kernel_size = int(abs((math.log(bands, 2) + b) / gamma))
        if kernel_size%2==0:
            kernel_size+=1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (batch, 1, rows, cols, bands)
        batch, _, bands, _, _ = x.shape
        gap = self.gap(x)  # [batch, 1, 1, 1, bands]
        gap = gap.view(batch, 1, bands)  # [batch, 1, bands]
        attn_weights = self.conv(gap)  # 滑动窗口计算局部光谱关系
        # Sigmoid 归一化到 [0,1]
        attn_weights = self.sigmoid(attn_weights)  # [batch, 1, bands]
        # 恢复形状为 (batch,1,1,1,bands)
        attn_weights = attn_weights.view(batch, 1, bands, 1, 1)
        return x * attn_weights

# ============其他论文编码器组件============
@register_encoder(name='HybridSN', dim=4, out_channels=256, feature_map=False)
class HybridSN_encoder(nn.Module):
  """code from: https://github.com/gokriznastic/HybridSN
  自适应输入维度"""
  def __init__(self, in_shape=None):
    super(HybridSN_encoder, self).__init__()
    bands, h, w = in_shape
    self.conv1 = nn.Conv3d(1, 8, (7, 3, 3))
    self.conv2 = nn.Conv3d(8, 16, (5, 3, 3))
    self.conv3 = nn.Conv3d(16, 32, (3, 3, 3))
    bands = bands - 12
    self.conv3_2d = nn.Conv2d(bands * 32, 64, (3,3))
    h = h - 8
    # 全连接层（256个节点）
    self.fc =  nn.Linear(h*h*64, 256)
    self.relu = nn.ReLU()

  def forward(self, x):
    if x.dim() == 4:
        x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
    elif x.dim() != 5:
        raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.relu(self.conv3(out))
    # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
    out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
    out = self.relu(self.conv3_2d(out))
    # flatten 操作，变为 18496 维的向量，
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out

class SPCModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SPCModuleIN, self).__init__()
                
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7,1,1), stride=(2,1,1), bias=False)
        #self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        
        input = input.unsqueeze(1)
        
        out = self.s1(input)
        
        return out.squeeze(1) 
    
class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(SPAModuleIN, self).__init__()
                
        # print('k=',k)
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,3,3), bias=False)
        #self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        # print(input.size())
        out = self.s1(input)
        out = out.squeeze(2)
        # print(out.size)
        
        return out
class ResSPC(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPC, self).__init__()
                
        self.spc1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm3d(in_channels),)
        
        self.spc2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),)
        
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
                
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))
        
        return F.leaky_relu(out + input)    
class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPA, self).__init__()
                
        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm2d(in_channels),)
        
        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))
        
        return F.leaky_relu(out + input)

@register_encoder(name='spec_transformer', dim=4, out_channels=get_base_configs()['hidden_size'], feature_map=False)
class SpecTransformer_encoder(nn.Module):
    def __init__(self, in_shape=None, config=get_base_configs(), global_pool=False):
        super(SpecTransformer_encoder, self).__init__()
        self.net = SpecTransformer(img_size=in_shape, config=config, global_pool=global_pool)
    def forward(self, x):
        return self.net(x)

# ================================================================================
# 对比学习
# ================================================================================
class Contrastive_Model(nn.Module):
    def __init__(self, encoder_model_name, in_shape=None):
        """
        encoder_model_name: str encoder模型的名称
        """
        super().__init__()
        items = ENCODER_REGISTRY[encoder_model_name] # 获取注册表中的模型构造信息
        encoder, dim, in_channels, feature_maps = items
        self.encoder = encoder(in_shape=in_shape)
        self.decoder = Contrastive_Decoder(in_channels, 128, mid_channels=128)
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