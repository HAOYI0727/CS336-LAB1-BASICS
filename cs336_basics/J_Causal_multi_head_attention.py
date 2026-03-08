# import torch
# import torch.nn as nn
# from einops import rearrange
# from cs336_basics.C_Linear import LinearModule
# from cs336_basics.I_Scaled_dot_product_attention import Scaled_dot_product_attention

# class CausalMultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_heads, device=None):
#         """
#         CausalMultiHeadAttention——因果多头注意力，将输入的稠密向量与输入的稠密向量进行点积来得到输出。
#         每个头的公式都是：out = softmax(QK^T / sqrt(d_k))V
#         参数：d_model (int): 输入的维度，也就是d_model
#              n_heads (int): 头的数量
#         """
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = n_heads
#         assert d_model % n_heads == 0, 'number of heads donen\' match d_model'
#         self.d_k = d_model // n_heads
        
#         self.wq = LinearModule(self.d_model, self.d_model)
#         self.wk = LinearModule(self.d_model, self.d_model)
#         self.wv = LinearModule(self.d_model, self.d_model)
#         self.wo = LinearModule(self.d_model, self.d_model)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, d_model = x.shape

#         q = self.wq(x)
#         k = self.wk(x)
#         v = self.wv(x)
        
#         q = rearrange(q, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
#         k = rearrange(k, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
#         v = rearrange(v, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        
#         # 创建因果掩码（布尔类型）
#         # 下三角为 True（允许关注），上三角为 False（不允许关注）
#         mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
#         mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
#         # 调用缩放点积注意力函数
#         attn_output = Scaled_dot_product_attention(q, k, v, mask)
        
#         attn_output = rearrange(attn_output, 'b n_h s d_k -> b s (n_h d_k)', n_h=self.num_heads)
#         out = self.wo(attn_output)
#         return out

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
from cs336_basics.C_Linear import LinearModule
from cs336_basics.I_Scaled_dot_product_attention import Scaled_dot_product_attention
from cs336_basics.G_RoPE import RotaryPositionalEmbedding

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = False, 
                 max_seq_len: Optional[int] = None, theta: float = 10000.0, device=None):
        """
        CausalMultiHeadAttention——因果多头注意力，支持是否使用RoPE位置编码
        参数:d_model (int): 输入的维度，也就是d_model
            n_heads (int): 头的数量
            use_rope (bool): 是否使用RoPE位置编码
            max_seq_len (Optional[int]): 最大序列长度，使用RoPE时需要
            theta (float): RoPE的底数超参数
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = n_heads
        self.use_rope = use_rope
        assert d_model % n_heads == 0, 'number of heads donen\' match d_model'
        self.d_k = d_model // n_heads
        
        if use_rope:
            assert self.d_k % 2 == 0, 'd_k must be even for RoPE'
            assert max_seq_len is not None, 'max_seq_len must be provided when use_rope=True'
        
        self.wq = LinearModule(self.d_model, self.d_model)
        self.wk = LinearModule(self.d_model, self.d_model)
        self.wv = LinearModule(self.d_model, self.d_model)
        self.wo = LinearModule(self.d_model, self.d_model)
        
        # RoPE位置编码 - 如果使用的话
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len=max_seq_len)
    
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:x: (batch_size, seq_len, d_model) 输入的稠密向量
            token_positions: (batch_size, seq_len) 每个token的位置信息，使用RoPE时需要
        返回值:out: (batch_size, seq_len, d_model) 输出的稠密向量
        """
        batch_size, seq_len, d_model = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = rearrange(q, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        k = rearrange(k, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        v = rearrange(v, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        
        # 如果使用RoPE，对每个头分别应用
        if self.use_rope:
            assert token_positions is not None, 'token_positions must be provided when use_rope=True'
            q_rope = []
            k_rope = []
            for head in range(self.num_heads):
                q_head = q[:, head, :, :]  # (batch_size, seq_len, d_k)
                k_head = k[:, head, :, :]  # (batch_size, seq_len, d_k)
                
                q_head_rope = self.rope(q_head, token_positions)
                k_head_rope = self.rope(k_head, token_positions)
                
                q_rope.append(q_head_rope.unsqueeze(1))
                k_rope.append(k_head_rope.unsqueeze(1))
            
            q = torch.cat(q_rope, dim=1)  # (batch_size, n_heads, seq_len, d_k)
            k = torch.cat(k_rope, dim=1)
        
        # 创建因果掩码（布尔类型）
        # 下三角为 True（允许关注），上三角为 False（不允许关注）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # 调用缩放点积注意力函数
        attn_output = Scaled_dot_product_attention(q, k, v, mask)
        
        attn_output = rearrange(attn_output, 'b n_h s d_k -> b s (n_h d_k)', n_h=self.num_heads)
        out = self.wo(attn_output)
        return out