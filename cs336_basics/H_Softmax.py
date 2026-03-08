import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    应用 softmax 操作到张量的指定维度上，返回一个概率分布。
    公式是: out = exp(x - x_max) / sum(exp(x - x_max)) 
    参数: x: (batch_size, seq_len, d_model) 输入的稠密向量
         dim: 归一化的维度
    返回值: out: (batch_size, seq_len, d_model) 归一化后的稠密向量，形状与输入相同，但指定维度变成概率分布
    """
    # 减去最大值以提高数值稳定性，防止 exp 过大导致溢出
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    # 计算 exp
    x_exp = torch.exp(x - x_max)
    # 求和并归一化
    x_softmax = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return x_softmax
