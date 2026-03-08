import torch

def gradient_clipping(parameters, max_l2_norm, epsilon=1e-6):
    """
    梯度裁剪函数 - 基于全局 L2 范数
    参数:parameters: 模型参数的可迭代对象
        max_l2_norm: 允许的最大 L2 范数
        epsilon: 防止除零的小常数
    返回:clip: 实际使用的裁剪系数（如果没有裁剪则返回 None）
    """
    # 收集所有非空梯度并计算全局范数
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    # 计算全局 L2 范数
    grad_l2 = torch.norm(torch.cat([grad.flatten() for grad in grads]), 2)
    
    # 需要裁剪时计算裁剪系数
    if grad_l2 > max_l2_norm:
        clip = max_l2_norm / (grad_l2 + epsilon)
        # 原地缩放所有梯度
        for grad in grads:
            grad.mul_(clip)