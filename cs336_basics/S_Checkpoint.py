import torch

def save_checkpoint(model, optimizer, iteration, out):
    # 构建检查点字典，使用字典可以方便地组织不同类型的状态
    checkpoint = {
        'model_state_dict': model.state_dict(),      # 模型权重
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态（动量、RMS等）
        'iteration': iteration,                       # 当前迭代次数
    }
    # 保存到文件
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    # 加载检查点文件
    checkpoint = torch.load(src)
    # 恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 获取保存时的迭代次数
    iteration = checkpoint['iteration']
    return iteration