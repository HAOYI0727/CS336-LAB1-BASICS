# import torch
# import torch.nn as nn
# from cs336_basics.C_Linear import LinearModule
# from cs336_basics.D_Embedding import EmbeddingModule
# from cs336_basics.E_RMSNorm import RMSNorm
# from cs336_basics.L_Transformer_block import TransformerBlock

# class TransformerLM(nn.Module):
#     """
#     TransformerLM——训练过程的封装，把包含Embedding、TransformerBlock、RMSNorm、LinearModule等组件包装在一起，形成一个完整的Transformer语言模型。
#     参数:vocab_size (int): 词表大小
#         context_length (int): 上下文长度
#         d_model (int): 输入的维度，也就是d_model
#         num_layers (int): 层数
#         num_heads (int): 头的数量
#         d_ff (int): 前馈神经网络的维度
#         rope_theta (float): 底数超参数
#         weights (dict[str, torch.Tensor], optional): 预训练权重，如果不提供则随机初始化
#     """
#     def __init__(self, vocab_size:int, context_length:int, d_model:int, 
#                  n_layers:int, n_heads:int, d_ff:int, 
#                  theta:float, weights:dict[str, torch.Tensor] = None, device=None):
#         super().__init__()
        
#         self.vocab_size = vocab_size
#         self.context_length = context_length
#         self.d_model = d_model
#         self.n_layers = n_layers
#         self.n_heads = n_heads
#         self.d_ff = d_ff
#         self.theta = theta
#         self.weights = weights
#         self.device = device
        
#         # 1. 创建Embedding层
#         self.embedding_module = EmbeddingModule(vocab_size, d_model, device)
        
#         # 2. 创建所有TransformerBlock层
#         self.transformer_blocks = nn.ModuleList()
#         for layer in range(n_layers):
#             if weights is not None:
#                 # 如果有预训练权重，使用它们
#                 transformer_block = TransformerBlock(
#                     d_model, n_heads, d_ff, context_length, theta,
#                     attn_q_proj_weight=weights[f"layers.{layer}.attn.q_proj.weight"],
#                     attn_k_proj_weight=weights[f"layers.{layer}.attn.k_proj.weight"],
#                     attn_v_proj_weight=weights[f"layers.{layer}.attn.v_proj.weight"],
#                     attn_o_proj_weight=weights[f"layers.{layer}.attn.output_proj.weight"],
#                     ln1_weight=weights[f"layers.{layer}.ln1.weight"],
#                     ln2_weight=weights[f"layers.{layer}.ln2.weight"],
#                     ffn_w1_weight=weights[f"layers.{layer}.ffn.w1.weight"],
#                     ffn_w2_weight=weights[f"layers.{layer}.ffn.w2.weight"],
#                     ffn_w3_weight=weights[f"layers.{layer}.ffn.w3.weight"]
#                 )
#             else:
#                 # 如果没有预训练权重，随机初始化
#                 transformer_block = TransformerBlock(
#                     d_model, n_heads, d_ff, context_length, theta
#                 )
#             self.transformer_blocks.append(transformer_block)
        
#         # 3. 创建最终的RMSNorm层
#         self.final_norm = RMSNorm(d_model, eps=1e-5)
        
#         # 4. 创建输出线性层
#         self.lm_head = LinearModule(d_model, vocab_size)
        
#         # 5. 如果有预训练权重，加载它们
#         if weights is not None:
#             self._load_weights(weights)
#         else:
#             # 可选：初始化权重
#             self._init_weights()
    
#     def _load_weights(self, weights: dict[str, torch.Tensor]):
#         """加载所有权重到对应的模块"""
#         # 加载embedding权重
#         self.embedding_module.embedding_matrix.data = weights["token_embeddings.weight"].data
#         # 加载final norm权重
#         self.final_norm.weight.data = weights["ln_final.weight"].data
#         # 加载lm_head权重
#         self.lm_head.weight.data = weights["lm_head.weight"].data
    
#     def _init_weights(self):
#         """初始化模型权重（可选）"""
#         # 可以在这里添加自定义的初始化逻辑
#         # 例如使用Xavier/Glorot初始化
#         for module in self.modules():
#             if isinstance(module, LinearModule):
#                 # 线性层初始化
#                 torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             elif isinstance(module, EmbeddingModule):
#                 # Embedding层初始化
#                 torch.nn.init.normal_(module.embedding_matrix, mean=0.0, std=0.02)
    
#     def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
#         """
#         参数: in_indices: (batch_size, seq_len) 输入的token索引
#         返回值: out_linear: (batch_size, seq_len, vocab_size) 输出的logits
#         """
#         # 1. Embedding
#         x = self.embedding_module(in_indices)  # (batch_size, seq_len, d_model)
        
#         # 2. 通过所有TransformerBlock层
#         for transformer_block in self.transformer_blocks:
#             x = transformer_block(x)
        
#         # 3. 最终的RMSNorm
#         x = self.final_norm(x)
        
#         # 4. 输出线性层（lm_head）
#         logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
#         # 注意：返回logits，不应用softmax（因为CrossEntropyLoss会内部应用log_softmax）
#         return logits

import torch
import torch.nn as nn
from typing import Optional, Literal
from cs336_basics.C_Linear import LinearModule
from cs336_basics.D_Embedding import EmbeddingModule
from cs336_basics.E_RMSNorm import RMSNorm
from cs336_basics.L_Transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    """
    TransformerLM——整个训练过程的封装，把包含Embedding、TransformerBlock、RMSNorm、LinearModule等组件包装在一起，形成一个完整的Transformer语言模型。
    参数:vocab_size (int): 词表大小
        context_length (int): 上下文长度
        d_model (int): 输入的维度，也就是d_model
        num_layers (int): 层数
        num_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        rope_theta (float): 底数超参数
        use_rmsnorm (bool): 是否使用RMSNorm（消融实验1）
        norm_position (Literal['pre', 'post']): 归一化位置（消融实验2）
        use_rope (bool): 是否使用RoPE位置编码（消融实验3）
        use_swiglu (bool): 是否使用SwiGLU（为True）或FFNSiLU（为False）（消融实验4）
        weights (dict[str, torch.Tensor], optional): 预训练权重
    """
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        d_model: int, 
        n_layers: int, 
        n_heads: int, 
        d_ff: int, 
        theta: float,
        use_rmsnorm: bool = True,
        norm_position: Literal['pre', 'post'] = 'pre',
        use_rope: bool = True,
        use_swiglu: bool = True,
        weights: dict[str, torch.Tensor] = None, 
        device=None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.theta = theta
        self.weights = weights
        self.device = device
        self.use_rmsnorm = use_rmsnorm
        self.norm_position = norm_position
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        
        # 1. 创建Embedding层
        self.embedding_module = EmbeddingModule(vocab_size, d_model, device)
        
        # 2. 创建所有TransformerBlock层
        self.transformer_blocks = nn.ModuleList()
        for layer in range(n_layers):
            if weights is not None:
                # 如果有预训练权重，使用它们
                transformer_block = TransformerBlock(
                    d_model, n_heads, d_ff, context_length, theta,
                    use_rmsnorm=use_rmsnorm,
                    norm_position=norm_position,
                    use_rope=use_rope,
                    use_swiglu=use_swiglu,
                    attn_q_proj_weight=weights.get(f"layers.{layer}.attn.q_proj.weight"),
                    attn_k_proj_weight=weights.get(f"layers.{layer}.attn.k_proj.weight"),
                    attn_v_proj_weight=weights.get(f"layers.{layer}.attn.v_proj.weight"),
                    attn_o_proj_weight=weights.get(f"layers.{layer}.attn.output_proj.weight"),
                    ln1_weight=weights.get(f"layers.{layer}.ln1.weight"),
                    ln2_weight=weights.get(f"layers.{layer}.ln2.weight"),
                    ffn_w1_weight=weights.get(f"layers.{layer}.ffn.w1.weight"),
                    ffn_w2_weight=weights.get(f"layers.{layer}.ffn.w2.weight"),
                    ffn_w3_weight=weights.get(f"layers.{layer}.ffn.w3.weight")
                )
            else:
                # 如果没有预训练权重，随机初始化
                transformer_block = TransformerBlock(
                    d_model, n_heads, d_ff, context_length, theta,
                    use_rmsnorm=use_rmsnorm,
                    norm_position=norm_position,
                    use_rope=use_rope,
                    use_swiglu=use_swiglu
                )
            self.transformer_blocks.append(transformer_block)
        
        # 3. 创建最终的归一化层（如果使用RMSNorm）
        if use_rmsnorm:
            self.final_norm = RMSNorm(d_model, eps=1e-5)
        else:
            self.final_norm = nn.Identity()
        
        # 4. 创建输出线性层
        self.lm_head = LinearModule(d_model, vocab_size)
        
        # 5. 如果有预训练权重，加载它们
        if weights is not None:
            self._load_weights(weights)
        else:
            # 可选：初始化权重
            self._init_weights()
    
    def _load_weights(self, weights: dict[str, torch.Tensor]):
        """加载所有权重到对应的模块"""
        # 加载embedding权重
        if "token_embeddings.weight" in weights:
            self.embedding_module.embedding_matrix.data = weights["token_embeddings.weight"].data
        # 加载final norm权重
        if "ln_final.weight" in weights and self.use_rmsnorm:
            self.final_norm.weight.data = weights["ln_final.weight"].data
        # 加载lm_head权重
        if "lm_head.weight" in weights:
            self.lm_head.weight.data = weights["lm_head.weight"].data
    
    def _init_weights(self):
        """初始化模型权重（可选）"""
        # 可以在这里添加自定义的初始化逻辑
        # 例如使用Xavier/Glorot初始化
        for module in self.modules():
            if isinstance(module, LinearModule):
                # 线性层初始化
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, EmbeddingModule):
                # Embedding层初始化
                torch.nn.init.normal_(module.embedding_matrix, mean=0.0, std=0.02)
    
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        参数: in_indices: (batch_size, seq_len) 输入的token索引
        返回值: out_linear: (batch_size, seq_len, vocab_size) 输出的logits
        """
        # 1. Embedding
        x = self.embedding_module(in_indices)  # (batch_size, seq_len, d_model)
        
        # 2. 通过所有TransformerBlock层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 3. 最终的归一化
        x = self.final_norm(x)
        
        # 4. 输出线性层（lm_head）
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        # 注意：返回logits，不应用softmax（因为CrossEntropyLoss会内部应用log_softmax）
        return logits