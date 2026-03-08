from torch import optim
import torch

class AdamW(optim.Optimizer):
    """
    实现 AdamW 优化算法——AdamW 通过将权重衰减与梯度更新解耦，修正了原始 Adam 中权重衰减的实现问题。
    更新规则如下：
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # 一阶矩（动量）更新
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        # 二阶矩（RMS）更新
    m_hat = m_t / (1 - beta1^t)                         # 一阶矩偏差修正
    v_hat = v_t / (1 - beta2^t)                         # 二阶矩偏差修正
    theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
    """
    
    def __init__(self, params, lr, betas, eps, weight_decay):
        """
        参数:params: 需要优化的参数或参数组
            lr: 学习率 (论文中的 alpha)
            betas: 梯度及其平方的移动平均系数 (beta1, beta2)
            eps: 分母中添加的小常数，提高数值稳定性
            weight_decay: 权重衰减系数 (论文中的 lambda)
        """
        # 将超参数存储到默认配置中，每个参数组都会继承这些默认值
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self):
        """
        执行单个优化步骤。
        """
        # 遍历每个参数组
        for group in self.param_groups:
            # 解包当前参数组的 beta 值
            beta1, beta2 = group['betas']
            
            # 遍历当前组中的所有参数
            for p in group['params']:
                # 如果参数没有梯度（例如冻结层），则跳过
                if p.grad is None:
                    continue
                # 获取该参数的梯度
                grad = p.grad
                # 获取该参数的状态字典，state 存储每个参数特有的变量，如动量缓冲区
                state = self.state[p]
                
                # 首次遇到该参数时初始化状态
                if len(state) == 0:
                    state['step'] = 0  # 记录更新步数，用于偏差修正
                    # 一阶矩估计（动量）- 跟踪梯度的移动平均
                    state['m'] = torch.zeros_like(p.data) 
                    # 二阶矩估计（RMS）- 跟踪梯度平方的移动平均
                    state['v'] = torch.zeros_like(p.data) 
                
                # 从状态中取出动量估计
                m, v = state['m'], state['v']
                # 步数加1（用于偏差修正）
                state['step'] += 1    

                # 更新动量: m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                # 更新 RMS: v = beta2 * v + (1 - beta2) * grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                # 计算偏差修正后的估计值
                bias_correction1 = 1 - beta1 ** state['step']  # 修正 m: 除以该因子
                bias_correction2 = 1 - beta2 ** state['step']  # 修正 v: 除以该因子
                # 偏差修正后的有效步长: lr / (1 - beta1^t)
                step_size = group['lr'] / bias_correction1 
                # 计算分母: sqrt(v_hat) + eps
                denom = v.sqrt().div_(bias_correction2 ** 0.5).add_(group['eps'])
                
                # 应用 Adam 更新（不包含权重衰减）——p = p - step_size * m / denom
                p.data.addcdiv_(m, denom, value=-step_size)

                # 应用解耦的权重衰减（AdamW 的核心创新）——p = p - lr * weight_decay * p
                # 与原始 Adam 将权重衰减合并到梯度中不同，AdamW 单独且直接地对参数应用权重衰减
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])