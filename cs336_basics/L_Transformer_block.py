# import torch
# import torch.nn as nn
# from cs336_basics.E_RMSNorm import RMSNorm
# from cs336_basics.F_SwiGLU import SwiGLU
# from cs336_basics.G_RoPE import RotaryPositionalEmbedding
# from cs336_basics.K_Causal_multi_head_attention_with_RoPE import CausalMultiHeadAttentionWithRoPE

# class TransformerBlock(nn.Module):
#     """
#     TransformerBlock——Transformer块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的Transformer块。
#     参数:d_model (int): 输入的维度，也就是d_model
#         n_heads (int): 头的数量
#         d_ffn (int): 前馈神经网络的维度
#         max_seq_len (int): 最大序列长度
#         theta (float): 底数超参数
#         attn_q_proj_weight (torch.Tensor, optional): 查询的权重，如果不提供则随机初始化
#         attn_k_proj_weight (torch.Tensor, optional): 键的权重
#         attn_v_proj_weight (torch.Tensor, optional): 值的权重
#         attn_o_proj_weight (torch.Tensor, optional): 输出的权重
#         ln1_weight (torch.Tensor, optional): 第一个LayerNorm的权重
#         ln2_weight (torch.Tensor, optional): 第二个LayerNorm的权重
#         ffn_w1_weight (torch.Tensor, optional): FFN第一个线性层的权重
#         ffn_w2_weight (torch.Tensor, optional): FFN第二个线性层的权重
#         ffn_w3_weight (torch.Tensor, optional): FFN第三个线性层的权重
#         device (str, optional): 设备
#     """
#     def __init__(self, d_model:int, n_heads:int, d_ff:int, max_seq_len:int, theta:float,
#                  attn_q_proj_weight:torch.Tensor = None, attn_k_proj_weight:torch.Tensor = None, attn_v_proj_weight:torch.Tensor = None, attn_o_proj_weight:torch.Tensor = None, 
#                  ln1_weight:torch.Tensor = None, ln2_weight:torch.Tensor = None, 
#                  ffn_w1_weight:torch.Tensor = None, ffn_w2_weight:torch.Tensor = None, ffn_w3_weight:torch.Tensor = None, device=None):
#         super(TransformerBlock, self).__init__()
        
#         # 模型基本参数
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_ff = d_ff
#         self.max_seq_len = max_seq_len
#         self.theta = theta
#         self.device = device
        
#         # 初始化均方根归一化
#         self.rms_norm1 = RMSNorm(d_model, eps=1e-5, device=device)
#         self.rms_norm2 = RMSNorm(d_model, eps=1e-5, device=device)
        
#         # 初始化SwiGLU
#         self.swiglu = SwiGLU(d_model, d_ff, device=device)
        
#         # 创建因果多头注意力模块
#         self.causal_multi_head_attention = CausalMultiHeadAttentionWithRoPE(
#             d_model, n_heads, max_seq_len, theta, device=device
#         )
        
#         # 如果有预训练权重，则加载它们
#         if attn_q_proj_weight is not None:
#             self.causal_multi_head_attention.wq.weight.data = attn_q_proj_weight.data
#         if attn_k_proj_weight is not None:
#             self.causal_multi_head_attention.wk.weight.data = attn_k_proj_weight.data
#         if attn_v_proj_weight is not None:
#             self.causal_multi_head_attention.wv.weight.data = attn_v_proj_weight.data
#         if attn_o_proj_weight is not None:
#             self.causal_multi_head_attention.wo.weight.data = attn_o_proj_weight.data
        
#         if ln1_weight is not None:
#             self.rms_norm1.weight.data = ln1_weight.data
#         if ln2_weight is not None:
#             self.rms_norm2.weight.data = ln2_weight.data
        
#         if ffn_w1_weight is not None:
#             self.swiglu.w1.weight.data = ffn_w1_weight.data
#         if ffn_w2_weight is not None:
#             self.swiglu.w2.weight.data = ffn_w2_weight.data
#         if ffn_w3_weight is not None:
#             self.swiglu.w3.weight.data = ffn_w3_weight.data

#     def forward(self, in_features: torch.Tensor):
#         """
#         参数: in_features: (batch_size, seq_len, d_model) 输入的张量
#         返回值: out: (batch_size, seq_len, d_model) 输出的张量
#         """
#         batch_size, seq_len, _ = in_features.shape
        
#         # 生成位置信息 - 需要扩展到batch维度
#         token_positions = torch.arange(seq_len, device=in_features.device)
#         token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
        
#         # 第一个残差块：LayerNorm -> Attention -> Add
#         x1 = self.rms_norm1(in_features)
#         x1 = self.causal_multi_head_attention(x1, token_positions)
#         x1 = x1 + in_features  # 残差连接
        
#         # 第二个残差块：LayerNorm -> FFN -> Add
#         x2 = self.rms_norm2(x1)
#         x2 = self.swiglu(x2)
#         out = x2 + x1  # 残差连接
        
#         return out

import torch
import torch.nn as nn
from typing import Optional, Literal
from cs336_basics.E_RMSNorm import RMSNorm
from cs336_basics.F_SwiGLU import SwiGLU
from cs336_basics.J_Causal_multi_head_attention import CausalMultiHeadAttention

class TransformerBlock(nn.Module):
    """
    TransformerBlock——Transformer块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的Transformer块。
    参数:d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
            - 当 use_swiglu=True 时，d_ff 应该是 8/3 * d_model（并确保能被64整除）
            - 当 use_swiglu=False 时，d_ff 应该是 4 * d_model（以匹配SwiGLU的参数量）
        max_seq_len (int): 最大序列长度
        theta (float): 底数超参数
        use_rmsnorm (bool): 是否使用RMSNorm（消融实验1）
        norm_position (Literal['pre', 'post']): 归一化位置（消融实验2）
        use_rope (bool): 是否使用RoPE位置编码（消融实验3）
        use_swiglu (bool): 是否使用SwiGLU（为True）或FFNSiLU（为False）（消融实验4）
        attn_q_proj_weight (torch.Tensor, optional): 查询的权重
        attn_k_proj_weight (torch.Tensor, optional): 键的权重
        attn_v_proj_weight (torch.Tensor, optional): 值的权重
        attn_o_proj_weight (torch.Tensor, optional): 输出的权重
        ln1_weight (torch.Tensor, optional): 第一个LayerNorm/RMSNorm的权重
        ln2_weight (torch.Tensor, optional): 第二个LayerNorm/RMSNorm的权重
        ffn_w1_weight (torch.Tensor, optional): FFN第一个线性层的权重
        ffn_w2_weight (torch.Tensor, optional): FFN第二个线性层的权重
        ffn_w3_weight (torch.Tensor, optional): FFN第三个线性层的权重（仅SwiGLU使用）
        device (str, optional): 设备
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        max_seq_len: int, 
        theta: float,
        use_rmsnorm: bool = True,
        norm_position: Literal['pre', 'post'] = 'pre',
        use_rope: bool = True,
        use_swiglu: bool = True,
        attn_q_proj_weight: torch.Tensor = None, 
        attn_k_proj_weight: torch.Tensor = None, 
        attn_v_proj_weight: torch.Tensor = None, 
        attn_o_proj_weight: torch.Tensor = None, 
        ln1_weight: torch.Tensor = None, 
        ln2_weight: torch.Tensor = None, 
        ffn_w1_weight: torch.Tensor = None, 
        ffn_w2_weight: torch.Tensor = None, 
        ffn_w3_weight: torch.Tensor = None, 
        device=None
    ):
        super(TransformerBlock, self).__init__()
        
        # 模型基本参数
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.use_rmsnorm = use_rmsnorm
        self.norm_position = norm_position
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        
        # 初始化归一化层（消融实验1和2）
        if use_rmsnorm:
            self.norm1 = RMSNorm(d_model, eps=1e-5, device=device)
            self.norm2 = RMSNorm(d_model, eps=1e-5, device=device)
        else:
            # 不使用RMSNorm意味着直接跳过归一化
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # 初始化前馈网络（消融实验4）- 使用修改后的SwiGLU类
        self.ffn = SwiGLU(d_model, d_ff, use_swiglu=use_swiglu, device=device)
        
        # 创建因果多头注意力模块（消融实验3）
        self.causal_multi_head_attention = CausalMultiHeadAttention(
            d_model, 
            n_heads, 
            use_rope=use_rope,
            max_seq_len=max_seq_len if use_rope else None,
            theta=theta,
            device=device
        )
        
        # 如果有预训练权重，则加载它们
        if attn_q_proj_weight is not None:
            self.causal_multi_head_attention.wq.weight.data = attn_q_proj_weight.data
        if attn_k_proj_weight is not None:
            self.causal_multi_head_attention.wk.weight.data = attn_k_proj_weight.data
        if attn_v_proj_weight is not None:
            self.causal_multi_head_attention.wv.weight.data = attn_v_proj_weight.data
        if attn_o_proj_weight is not None:
            self.causal_multi_head_attention.wo.weight.data = attn_o_proj_weight.data
        
        if ln1_weight is not None and use_rmsnorm:
            self.norm1.weight.data = ln1_weight.data
        if ln2_weight is not None and use_rmsnorm:
            self.norm2.weight.data = ln2_weight.data
        
        # 加载FFN权重 - 根据use_swiglu决定加载哪些权重
        if ffn_w1_weight is not None:
            self.ffn.w1.weight.data = ffn_w1_weight.data
        if ffn_w2_weight is not None:
            self.ffn.w2.weight.data = ffn_w2_weight.data
        if use_swiglu and ffn_w3_weight is not None:
            self.ffn.w3.weight.data = ffn_w3_weight.data

    def forward(self, in_features: torch.Tensor):
        """
        参数: in_features: (batch_size, seq_len, d_model) 输入的张量
        返回值: out: (batch_size, seq_len, d_model) 输出的张量
        """
        batch_size, seq_len, _ = in_features.shape
        
        # 生成位置信息 - 需要扩展到batch维度
        token_positions = torch.arange(seq_len, device=in_features.device)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
        
        if self.norm_position == 'pre':
            # 预归一化（原始Transformer架构）
            # 第一个残差块
            x1 = self.norm1(in_features)
            x1 = self.causal_multi_head_attention(x1, token_positions if self.use_rope else None)
            x1 = x1 + in_features  # 残差连接
            
            # 第二个残差块
            x2 = self.norm2(x1)
            x2 = self.ffn(x2)
            out = x2 + x1  # 残差连接
            
        elif self.norm_position == 'post':
            # 后归一化（原始论文中的架构）
            # 第一个残差块：先Attention再加归一化
            x1 = self.causal_multi_head_attention(in_features, token_positions if self.use_rope else None)
            x1 = self.norm1(x1 + in_features)  # 残差连接后归一化
            
            # 第二个残差块
            x2 = self.ffn(x1)
            out = self.norm2(x2 + x1)  # 残差连接后归一化
        
        return out