import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
from cs336_basics.C_Linear import LinearModule
from cs336_basics.I_Scaled_dot_product_attention import Scaled_dot_product_attention
from cs336_basics.G_RoPE import RotaryPositionalEmbedding

class CausalMultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, theta: float, device=None):
        """
        带RoPE的因果多头注意力
        参数:d_model (int): 输入的维度
            n_heads (int): 头的数量
            max_seq_len (int): 最大序列长度
            theta (float): RoPE的底数超参数
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = n_heads
        assert d_model % n_heads == 0, 'number of heads donen\' match d_model'
        self.d_k = d_model // n_heads
        assert self.d_k % 2 == 0, 'd_k must be even for RoPE'
        
        # 线性投影层
        self.wq = LinearModule(self.d_model, self.d_model)
        self.wk = LinearModule(self.d_model, self.d_model)
        self.wv = LinearModule(self.d_model, self.d_model)
        self.wo = LinearModule(self.d_model, self.d_model)
        
        # RoPE位置编码 - 对每个头的d_k维度应用
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len=max_seq_len)
        
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:x: (batch_size, seq_len, d_model) 输入的稠密向量
            token_positions: (batch_size, seq_len) 每个token的位置信息
        返回值:out: (batch_size, seq_len, d_model) 输出的稠密向量
        """
        batch_size, seq_len, d_model = x.shape

        # QKV投影
        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)
        v = self.wv(x)
        
        # 分割多头
        q = rearrange(q, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        k = rearrange(k, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        v = rearrange(v, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        
        # 对每个头分别应用RoPE
        # q, k shape: (batch_size, n_heads, seq_len, d_k)
        q_rope = []
        k_rope = []
        for head in range(self.num_heads):
            # 对每个头的q和k应用RoPE
            q_head = q[:, head, :, :]  # (batch_size, seq_len, d_k)
            k_head = k[:, head, :, :]  # (batch_size, seq_len, d_k)
            
            q_head_rope = self.rope(q_head, token_positions)
            k_head_rope = self.rope(k_head, token_positions)
            
            q_rope.append(q_head_rope.unsqueeze(1))
            k_rope.append(k_head_rope.unsqueeze(1))
        
        q = torch.cat(q_rope, dim=1)  # (batch_size, n_heads, seq_len, d_k)
        k = torch.cat(k_rope, dim=1)
        
        # 创建因果掩码（布尔类型，因为Scaled_dot_product_attention需要这种格式）
        # 下三角为True（允许关注），上三角为False（不允许关注）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # 调用缩放点积注意力函数
        attn_output = Scaled_dot_product_attention(q, k, v, mask)
        
        # 合并多头
        attn_output = rearrange(attn_output, 'b n_h s d_k -> b s (n_h d_k)', n_h=self.num_heads)
        
        # 输出投影
        out = self.wo(attn_output)
        return out