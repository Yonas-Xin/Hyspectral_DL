from classifier_model.Models.Encoder import *
from classifier_model.Models.Decoder import *
import torch.nn as nn
MODEL_REGISTRY = {}

def register_model(name: str | None = None):
    """模型注册装饰器, name缺省则用类/函数的 __name__"""
    def wrapper(obj):
        key = name or obj.__name__
        if key in MODEL_REGISTRY:
            raise ValueError(f"模型名称重复: {key}")
        MODEL_REGISTRY[key] = obj
        return obj
    return wrapper

class My_Model(nn.Module):
    def __init__(self, out_classes=None, in_shape=None): # 框架需要两个输入
        super().__init__()
        self.lock_grad = False
        self.unfreeze_idx = None

    def freeze_encoder(self):
        if self.lock_grad:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print('The encoder parameters have been frozen.')
        else:
            print('Attention: The encoder parameters are not frozen! Cause: Too many parameters do not match!')
    
    def if_draw_feature_map(self):
        return True

    def load_encoder_params(self, state_dict):
        try:
            self.encoder.load_state_dict(state_dict, strict=True)
            self.lock_grad = True
            print("The encoder weights match exactly! Parameters loaded successfully.")
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
            if skipped == 0:
                self.lock_grad = True
            else:
                print(f"Atteneion: The encoder weights do not match exactly! Skipped loading these keys: {skipped}")
                if len(skipped) <= 4:
                    self.lock_grad = True
                else:
                    self.lock_grad = False
                    print(f"{len(skipped)} parameters are skipped, Please check if the pre-trained model is consistent with the parameters of the training model. ")

@register_model()
class SRACN(My_Model):
    def __init__(self, out_classes, in_shape=None):
        super().__init__()
        if in_shape is None:
            raise ValueError("in_shape must be provided for the model.")
        self.encoder = SRACN_Encoder(in_shape=in_shape)
        self.decoder = deep_classfier(128, out_classes, mid_channels=1024)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

@register_model()
class HybridSN(My_Model):
    """code from: https://github.com/gokriznastic/HybridSN
    自适应输入维度"""
    def __init__(self, out_classes, in_shape=None):
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
        x = x.flatten(1)
        x = self.decoder(x)
        return x

@register_model()
class spec_transformer(My_Model):
    def __init__(self, out_classes, in_shape=None):
        super().__init__()
        if in_shape is None:
            raise ValueError("in_shape must be provided for the model.")
        self.encoder = SpecTransformer_encoder(in_shape)
        self.decoder = nn.Linear(self.encoder.net.hidden_size, out_classes)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        elif x.dim() == 5:
            x = x.squeeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def if_draw_feature_map(self):
        return False

# ================================================================================
# 对比学习
# ================================================================================
class Ete_Model(nn.Module):
    def __init__(self,
                 encoder_model_name : str,
                 in_shape : tuple = None, 
                 K: int = 65536, # 为了与Moco初始化对应，这里不管
                 m: float = 0.999,
                 T=0.07):
        super().__init__()  
        self.encoder_q = Contrastive_Model(encoder_model_name=encoder_model_name, in_shape=in_shape)
        self.encoder_k = Contrastive_Model(encoder_model_name=encoder_model_name, in_shape=in_shape)
        self.T = T
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # 确保k与q的初始参数一致，k模型不反向传播参数

    @torch.no_grad() # 调用函数时不计算梯度， 外部调用， 确保多卡正常训练
    def update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder: k的参数复制
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_q.data # 更新k参数

    def calculate_logits(self, q, k): # 计算logits和labels, 确保多卡训练， 外部调用
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            k = nn.functional.normalize(k, dim=1)
        similarity_matrix = torch.matmul(q, k.T)
        mask = torch.eye(q.shape[0], dtype=torch.bool).to(q.device)
        positives = similarity_matrix[mask].view(q.shape[0], -1) # 对角线位置为正样本对
        negatives = similarity_matrix[~mask].view(q.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)  # 拼接
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        return logits, labels

    def forward(self, input_q, input_k):
        q = self.encoder_q(input_q)
        with torch.no_grad():
            k = self.encoder_k(input_k)
        return q, k

class Moco_Model(nn.Module): # 单GPU训练的Moco框架
    def __init__(self, 
                 encoder_model_name: str, 
                 in_shape: tuple = None,
                 K: int = 65536,
                 m: float = 0.999,
                 T: float = 0.7):
        super().__init__()
        self.K = K
        self.T = T
        self.m = m

        self.encoder_q = Contrastive_Model(encoder_model_name=encoder_model_name, in_shape=in_shape)
        self.encoder_k = Contrastive_Model(encoder_model_name=encoder_model_name, in_shape=in_shape)
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # 确保k与q的初始参数一致，k模型不反向传播参数

        # create the queue
        self.register_buffer("queue", torch.randn(1024, K)) # 这里填模型的decoder输出维度
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder: k的参数动量更新
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def calculate_logits(self, q, k): # 计算logits和labels, 确保多卡训练， 外部调用
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            k = nn.functional.normalize(k, dim=1)
                # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1) # 计算批次中每每两个正样本对的相似度
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()]) # 计算批次中样本与队列中所有负样本的相似度

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        return logits, labels

    def forward(self, input_q, input_k):
        """
        Input:
            input_q: a batch of query images
            input_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(input_q)  # queries: NxC
        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.encoder_k(input_k)  # keys: NxC
        return q, k