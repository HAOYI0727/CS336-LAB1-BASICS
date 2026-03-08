import torch
from torch import nn
import math
from einops import einsum

class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None, bias=False):
        """
        自定义线性层，不包含偏置项。
        参数:
            in_features (int): 输入特征的数量 (d_in)。
            out_features (int): 输出特征的数量 (d_out)。
            device (torch.device, optional): 存储参数的设备。
            dtype (torch.dtype, optional): 参数的数据类型。
        """
        super().__init__() # 调用父类构造函数
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        # 创建权重矩阵 (d_out, d_in)，注意 PyTorch 标准是 (out, in)
        weight_shape = (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(weight_shape, device=device, dtype=dtype))
        
        # 初始化权重: N(μ=0, σ² = 2 / (din + dout)), truncated at [-3σ, 3σ]
        sigma = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... d_in,  d_out d_in -> ... d_out')