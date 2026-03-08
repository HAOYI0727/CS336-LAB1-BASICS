import torch
from torch import nn

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        自定义嵌入层
        参数:
            num_embeddings (int): 词汇表的大小（vocab_size）。
            embedding_dim (int): 嵌入向量的维度 (d_model)。
            device (torch.device, optional): 存储参数的设备。
            dtype (torch.dtype, optional): 参数的数据类型。
        """
        super().__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        # 创建嵌入矩阵 (num_embeddings, embedding_dim)
        embedding_shape = (num_embeddings, embedding_dim)
        self.embedding_matrix = nn.Parameter(torch.empty(embedding_shape, device=device, dtype=dtype))
        
        std = 1.0
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids] # 从词表到词向量的映射的神经网络里读取token_ids输出词向量