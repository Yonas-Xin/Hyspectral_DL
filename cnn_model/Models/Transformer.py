import math
import copy
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Linear, LayerNorm

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

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
        attention_output = self.out(context_layer)  # 再经过一个全连接层MLP将数据再次汇总
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
        x = x + h  # 残差连接

        h = x
        x = self.ffn_norm(x)  # 全连接
        x = self.ffn(x)  # 全连接
        x = x + h
        return x
    
class Transformer(nn.Module):
    """
    Transformer
    
    Parameters:
    config: dict, Transformer的配置参数

    数据流:
    输入: x (batch_size, seq_length, hidden_size)
    输出: x (batch_size, seq_length, hidden_size)
    """
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config['hidden_size'], eps=1e-6)  # 层归一化
        for _ in range(config['num_layers']):  # 构造多个 Transformer 层
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))  # 深拷贝

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)
        x = self.encoder_norm(x)
        return x