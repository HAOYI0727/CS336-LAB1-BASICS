import torch
import torch.nn as nn
     
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        RoPE——旋转位置编码，它通过将输入的稠密向量旋转来稳定训练。
        参数:
            theta (float): 底数超参数，控制旋转频率
            d_k (int): 输入的维度，即d_model，必须是偶数
            max_seq_len (int): 最大序列长度
            device (torch.device): 设备
        """
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        #计算频率
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        #记录每个token的位置信息
        positions = torch.arange(self.max_seq_len)
        #计算正弦和余弦
        angles = torch.outer(positions, freqs) #outer是外积，即每个位置都与每个频率相乘 shape: [max_seq_len, d_k//2]
        self.register_buffer("cos", torch.cos(angles), persistent=False) 
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''    
        公式是: out = x * cos(theta * position) - x * sin(theta * position)
        输入:
            x: (batch_size, seq_len, d_model) 输入的稠密向量
            token_positions: (batch_size, seq_len) 每个token的位置信息
        输出:
            out: (batch_size, seq_len, d_model) 输出的稠密向量
        '''
        # 1. 根据token的位置，找到对应的旋转角度（用cos和sin表示）
        cos = self.cos[token_positions]  # [batch, seq, d//2]
        sin = self.sin[token_positions]  # [batch, seq, d//2]
        
        # 2. 把输入向量重新排列，方便处理
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, d_model // 2, 2)
        
        # 3. 对输入向量进行旋转
        rotated = torch.stack([
            x[..., 0] * cos - x[..., 1] * sin,  # 新的偶数部分
            x[..., 0] * sin + x[..., 1] * cos   # 新的奇数部分
        ], dim=-1)  # [batch, seq, d//2, 2]
        
        return rotated.flatten(-2)  # [batch, seq, d]