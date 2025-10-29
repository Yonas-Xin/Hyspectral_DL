import math
import copy
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm

def get_configs():
    configs = {
        'hidden_size': 768,
        'split_size': 10, # 每个patch在通道维度的大小
        'num_heads': 12,
        'num_layers': 12,
        'attention_dropout_rate': 0.0,
        'dropout_rate': 0.1,
    }
    return configs

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def make_channels_divisible(x, divisor): # 将通道数调整为divisor的整数倍
    if len(x.shape) == 4:
        _, C, H, W = x.shape
    else:
        C, H, W = x.shape
    remainder = C % divisor
    if remainder == 0:
        return x
    pad_channels = divisor - remainder
    return nn.ConstantPad2d((0, 0, 0, 0, 0, pad_channels), 0)(x)

def make_divisible(x, divisor): # 将x调整为divisor的整数倍
    remainder = x % divisor
    if remainder == 0:
        return x
    return x + divisor - remainder


class Embeddings(nn.Module):
    """ Construct the embeddings from patch, position embeddings. """

    def __init__(self, img_size, config):
        super(Embeddings, self).__init__()
        self.hybrid = None
        C, H, W = img_size
        self.split_size = config['split_size']  # 每个补丁在通道维度的大小
        C = make_divisible(C, self.split_size)  # 保证通道数是10的整数倍
        n_patches = (C // self.split_size) # 从通道维度切分补丁

        # 将图像切分为若干补丁并映射到指定的隐藏空间
        self.patch_embeddings = Conv3d(in_channels=1,
                                       out_channels=config['hidden_size'],
                                       kernel_size=(self.split_size, H, W),  # 等于 Patch 的大小，确保每个 Patch 的信息被映射到一个嵌入向量中
                                       stride=(self.split_size, H, W))  # 等于 Patch 的大小，保证不重叠分割
        # 提供序列位置信息，用于帮助 Transformer 区分不同的 Patch 顺序.  n_patches + 1：包含 CLS Token 的位置
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config['hidden_size']))  # 绝对位置
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))

        self.dropout = Dropout(config['attention_dropout_rate'])

    def forward(self, x):
        x = make_channels_divisible(x, self.split_size) # 保证通道数是10的整数倍
        B = x.shape[0]  # x.shape (b, c, w, h)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, hidden_size) -> (b, 1, hidden_size), 用来分类的标记
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # (b, c, w, h) -> (b, c, 1, w, h) 变成5维以适应Conv3d
        x = self.patch_embeddings(x)  # x.shape (b, hidden_size, w/patch_size, h/patch_size)
        x = x.flatten(2)  # x.shape (b, hidden_size, w/patch_size*h/patch_size) 下面用N表示w/patch_size*h/patch_size
        x = x.transpose(-1, -2)  # x.shape (b, N, hidden_size)
        x = torch.cat((cls_tokens, x), dim=1)  # x.shape (b, N+1, hidden_size)

        embeddings = x + self.position_embeddings  # 加上位置信息
        embeddings = self.dropout(embeddings)
        return embeddings

class Attention(nn.Module): # 多头自注意力机制
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config['num_heads']
        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config['hidden_size'], self.all_head_size)  # 当前位置的查询向量
        self.key = Linear(config['hidden_size'], self.all_head_size)  # 所有位置的键向量
        self.value = Linear(config['hidden_size'], self.all_head_size)  # 所有位置的值向量

        self.out = Linear(config['hidden_size'], config['hidden_size'])
        self.attn_dropout = Dropout(config['attention_dropout_rate'])
        self.proj_dropout = Dropout(config['attention_dropout_rate'])

        self.softmax = Softmax(dim=-1)  # 计算注意力权重，通过对 Query 和 Key 的点积结果进行归一化

    def transpose_for_scores(self, x):
        # eg: 768 -> 12*64 将数据转换为适合多头注意力操作的形状，以便能够并行处理多个注意力头
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # 将最后一维(hidden_dim)改为(num_attention_heads, attention_head_size)
        return x.permute(0, 2, 1, 3)  # x.shape (b, num_attention_heads,N+1,attention_head_size)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # Linear(in_features=768, out_features=768, bias=True)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # 注意力分数 = query 点积 key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 序列越长分数越大，对其进行比例缩放
        attention_probs = self.softmax(attention_scores)  # 得到实际权重值
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # 把权重分配给value 每个位置信息从这一步开始考虑全局特征
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 将12头注意力还原回原来维度
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)  # 再经过一个全连接层MLP将数据再次汇总 (你问我为什么，经验所得)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config['hidden_size'], config['hidden_size'] * 4)
        self.fc2 = Linear(config['hidden_size'] * 4, config['hidden_size'])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config['dropout_rate'])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config['hidden_size']
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)  # 标准化
        x = self.attn(x)
        x = x + h  # 想象成残差连接

        h = x
        x = self.ffn_norm(x)  # 全连接
        x = self.ffn(x)  # 全连接
        x = x + h
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config['hidden_size'], eps=1e-6)  # 归一化(和BN不一样哦)
        for _ in range(config['num_layers']):  # 构造多个 Transformer 层
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))  # 深拷贝

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded
    
class Transformer(nn.Module):
    def __init__(self, img_size, config):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size, config)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded
    

class SpecTransformer(nn.Module):
    def __init__(self, img_size, config=get_configs()):
        super(SpecTransformer, self).__init__()
        self.transformer = Transformer(img_size, config)
        self.hidden_size = config['hidden_size']

    def forward(self, x):
        x = self.transformer(x)
        return x[:, 0]  # 直接返回CLS token的编码
        
if __name__ == "__main__":
    x = torch.randn(2, 166, 64, 64)
    model = SpecTransformer(img_size=(166, 64, 64))
    x = model(x)
    print(x.shape)