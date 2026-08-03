import torch

class SpectralL2NormPerPixel:
    """
    对每个空间像元的光谱向量进行 L2 归一化

    输入:
      - (bands,) 或 (bands, H, W)
    输出:
      - 同形状
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, image):
        """
        image: np.ndarray 或 torch.Tensor
        """
        if not torch.is_tensor(image):
            x = torch.from_numpy(image).float()
        else:
            x = image.float()
        if x.dim() == 1:
            norm = torch.norm(x, p=2)
            if norm < self.eps:
                return x
            return x / norm
        if x.dim() == 3:
            # 对 band 维度做 L2
            # norm shape: (1, H, W)
            norm = torch.norm(x, p=2, dim=0, keepdim=True)
            # 防止除 0
            norm_safe = torch.clamp(norm, min=self.eps)
            x_norm = x / norm_safe
            # 对原本近 0 的像元，保持原值
            zero_mask = norm < self.eps
            if zero_mask.any():
                x_norm = torch.where(zero_mask.expand_as(x), x, x_norm)

            return x_norm
        raise ValueError(
            f"Unsupported input shape {tuple(x.shape)}, "
            "expect (bands,) or (bands, H, W)."
        )