import numpy as np
import torch

def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集中采样一个批次的数据
    参数:dataset: 一维numpy数组，包含完整的token ID序列
        batch_size: 批次大小
        context_length: 上下文长度（输入序列长度）
        device: 目标设备
    返回值:(inputs, targets)元组，每个都是形状为(batch_size, context_length)的torch张量
    """
    # 获取数据集长度
    data_length = len(dataset)
    
    # 计算最大起始索引
    max_start_idx = data_length - context_length - 1
    
    # 1. 随机选择起始位置
    start_indices = np.random.randint(0, max_start_idx + 1, size=(batch_size,))
    
    # 2. 为每个起始位置创建连续的索引序列
    # start_indices[:, None] 将形状从 (batch_size,) 变为 (batch_size, 1)
    # 结果形状: (batch_size, context_length + 1)
    idx = start_indices[:, None] + np.arange(context_length + 1)
    
    # 3. 根据索引从数据集中获取实际的token
    tokens = dataset[idx]
    
    # 4. 分离输入和目标
    inputs = tokens[:, :-1]  # 取每个序列的前context_length个token（去掉最后一列）
    targets = tokens[:, 1:]  # 取每个序列的后context_length个token（去掉第一列）
    
    # 5. 转换为PyTorch张量并移动到指定设备
    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(targets).long().to(device)
    )