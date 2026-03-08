import math

def cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    """
    余弦退火学习率调度函数（带预热阶段）
    参数:it: 当前迭代次数
        max_lr: 最大学习率（预热结束时的峰值）
        min_lr: 最小学习率（余弦衰减结束时的谷值）
        warmup_iters: 预热迭代次数
        cosine_cycle_iters: 余弦衰减的总迭代次数（包含预热）
    返回:当前迭代应该使用的学习率
    """
    # 阶段1: 预热阶段 - 线性增长
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 阶段3: 超过余弦周期 - 保持最小学习率
    elif it > cosine_cycle_iters:
        return min_lr
    # 阶段2: 余弦衰减阶段 - 从峰值下降到谷值
    else:
        # 计算当前在余弦周期中的进度 (0 到 1)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # 余弦因子: 从1逐渐降到0
        cosine = (1 + math.cos(math.pi * progress)) / 2
        # 在最大和最小学习率之间插值
        return min_lr + (max_lr - min_lr) * cosine