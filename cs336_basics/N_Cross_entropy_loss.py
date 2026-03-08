import torch

def cross_entropy_loss(inputs, targets):
    """
    计算交叉熵损失
    参数:inputs: 模型输出的logits，形状 (batch_size, num_classes)
        targets: 目标类别索引，形状 (batch_size,)
    返回值:标量损失值（批次平均值）
    """
    batch_size = inputs.shape[0]
    
    # 1. 数值稳定性：减去最大值
    inputs_max = torch.max(inputs, dim=-1, keepdim=True)[0]
    inputs_stable = inputs - inputs_max
    
    # 2. 抵消 log 和 exp：直接计算 log_softmax
    # log_softmax = x - log(sum(exp(x)))
    log_sum_exp = torch.logsumexp(inputs_stable, dim=-1, keepdim=True)
    log_probs = inputs_stable - log_sum_exp  # 这就是 log_softmax
    
    # 3. 提取目标类别的对数概率
    target_log_probs = log_probs[torch.arange(batch_size), targets]
    
    # 4. 负对数似然（已经是 -log(prob)）
    losses = -target_log_probs
    
    # 5. 返回批次平均值
    return losses.mean()