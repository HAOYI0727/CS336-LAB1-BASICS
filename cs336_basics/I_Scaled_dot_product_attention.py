import torch
import torch.nn as nn

def Scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    """
    前向传播
    参数：
        Q->query: (batch_size, num_heads, query_len, d_k)
        K->key:   (batch_size, num_heads, key_len, d_k)
        V->value: (batch_size, num_heads, key_len, d_v)
        mask:  (batch_size, 1, query_len, key_len)
    """
    # 
    d_k = Q.size(-1)
    # 计算注意力分数
    scores = torch.einsum('... q d, ... k d -> ... q k', Q, K) / (d_k ** 0.5)
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    # softmax
    attention_weights = torch.softmax(scores, dim=-1)
    # 加权求和
    output = torch.einsum('... q k, ... k v -> ... q v', attention_weights, V)
    return output