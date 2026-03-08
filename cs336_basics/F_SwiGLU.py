import torch
import torch.nn as nn
from cs336_basics.C_Linear import LinearModule

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, use_swiglu=True, device=None):
        '''
        SwiGLU/FFNSiLU 前馈网络模块，支持两种结构：
        1. SwiGLU: 三个权重矩阵，公式: out = w2(silu(w1(x)) * w3(x))
        2. FFNSiLU: 两个权重矩阵，公式: out = w2(silu(w1(x)))
        
        参数：
            d_model（int）: 输入的维度
            d_ff（int）: 前馈网络的隐藏层维度
                - 当 use_swiglu=True 时，d_ff 应该是 8/3 * d_model（并确保能被64整除）
                - 当 use_swiglu=False 时，d_ff 应该是 4 * d_model（以匹配SwiGLU的参数量）
            use_swiglu（bool）: 是否使用SwiGLU结构（True）或FFNSiLU结构（False）
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_swiglu = use_swiglu
        
        # 共享的线性层
        self.w1 = LinearModule(d_model, d_ff, bias=False)
        self.w2 = LinearModule(d_ff, d_model, bias=False)
        
        # SwiGLU特有的第三个线性层
        if use_swiglu:
            self.w3 = LinearModule(d_model, d_ff, bias=False)

    def silu(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        '''
        公式:
            - SwiGLU: out = w2(silu(w1(x)) * w3(x))
            - FFNSiLU: out = w2(silu(w1(x)))
        '''
        if self.use_swiglu:
            # SwiGLU: 三个权重矩阵
            return self.w2(self.silu(self.w1(x)) * self.w3(x))
        else:
            # FFNSiLU: 两个权重矩阵
            return self.w2(self.silu(self.w1(x)))