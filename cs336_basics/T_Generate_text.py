import argparse
import torch
import numpy as np
from typing import List, Optional, Union
import sys
import os

# 将项目根目录添加到Python路径，以便我们可以导入自己的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.M_Transformer_LM import TransformerLM
from cs336_basics.S_Checkpoint import load_checkpoint
from cs336_basics.O_AdamW import AdamW


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    对logits应用温度缩放和softmax。
    参数:logits: 原始logits张量，形状为 (..., vocab_size)
        temperature: 用于缩放的温度参数。较低的值使分布更尖锐。
    返回:经过温度缩放的softmax概率
    """
    # 应用温度缩放
    scaled_logits = logits / temperature
    
    # 为数值稳定性应用softmax，减去最大值以提高数值稳定性
    max_logits = torch.max(scaled_logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(scaled_logits - max_logits)
    probabilities = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    return probabilities


def top_p_sampling(probabilities: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    对概率分布应用top-p（核）采样。
    参数:probabilities: 概率分布，形状为 (..., vocab_size)
        p: 核采样的累积概率阈值
    返回:修改后的概率分布，低概率的token被屏蔽
    """
    # 按降序排序概率
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 创建需要保留的token的掩码（累积概率 <= p）
    mask = cumulative_probs <= p
    # 始终至少保留最高概率的token
    mask[..., 0] = True
    
    # 将不在核中的token的概率设为零
    filtered_probs = sorted_probs * mask.float()
    # 重新归一化
    filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)
    # 创建与输入相同形状的输出张量
    output_probs = torch.zeros_like(probabilities)
    output_probs.scatter_(-1, sorted_indices, filtered_probs)
    
    return output_probs


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
    eos_token: Optional[str] = "<|endoftext|>"
) -> str:
    """
    使用训练好的语言模型生成文本。
    参数:model: 训练好的Transformer语言模型
        tokenizer: 用于编码/解码文本的分词器
        prompt: 输入的提示字符串
        max_tokens: 要生成的最大token数量
        temperature: 采样温度（越低越确定）
        top_p: 核采样的top-p阈值
        device: 运行生成的设备
        eos_token: 序列结束标记
    返回:生成的文本字符串
    """
    model.eval()
    
    # 编码提示文本
    prompt_tokens = tokenizer.encode(prompt)
    # 转换为张量并移动到指定设备
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    # 逐个生成token
    generated_tokens = prompt_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 通过模型前向传播
            logits = model(input_ids)
            # 获取最后一个token的logits
            next_token_logits = logits[0, -1, :]  # 形状: (vocab_size,)
            
            # 应用温度缩放和softmax
            probabilities = softmax_with_temperature(next_token_logits, temperature)
            
            # 如果指定了top-p，则应用核采样
            if top_p < 1.0:
                probabilities = top_p_sampling(probabilities, top_p)
            
            # 采样下一个token
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            # 添加到生成的序列中
            generated_tokens.append(next_token)
            
            # 检查是否遇到序列结束标记
            if eos_token is not None:
                decoded_token = tokenizer.decode([next_token])
                if decoded_token.strip() == eos_token.strip():
                    break
            
            # 更新下一次迭代的输入
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            # 如果序列太长则截断（避免内存问题）
            if input_ids.size(1) > model.context_length:
                input_ids = input_ids[:, -model.context_length:]
    
    # 解码生成的token
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def load_model_and_tokenizer(checkpoint_path: str, vocab_path: str, merges_path: str, device: str = "cpu"):
    """
    从检查点和词汇文件加载训练好的模型和分词器。
    参数:checkpoint_path: 模型检查点路径
        vocab_path: 分词器词汇表路径
        merges_path: 分词器合并规则路径
        device: 加载模型的设备
    返回:(model, tokenizer)的元组
    """
    # 导入分词器
    try:
        from cs336_basics.B_Tokenizer import Tokenizer
        tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    except ImportError:
        # 如果分词器在其他位置，则回退处理
        print("警告: 无法导入分词器。您可能需要实现或调整导入路径。")
        tokenizer = None
    
    # 加载检查点以获取模型配置
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取模型超参数 - 处理不同的检查点格式
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 默认配置 - 根据需要调整
        config = {
            'vocab_size': 18017,
            'context_length': 256,
            'n_layers': 4,
            'd_model': 512,
            'n_heads': 16,
            'theta': 10000.0,
            'd_ff': 1344,
            # 消融实验参数默认值
            'use_rmsnorm': True,
            'norm_position': 'pre',
            'use_rope': True,
            'use_swiglu': True
        }
        print("警告: 使用默认模型配置。请根据需要调整。")
    
    # 确保参数名称与 TransformerLM 的 __init__ 匹配
    # 映射可能的参数名称变体
    param_mapping = {
        'n_layers': 'n_layers',
        'num_layers': 'n_layers',
        'layers': 'n_layers',
        'n_heads': 'n_heads',
        'num_heads': 'n_heads',
        'heads': 'n_heads',
        'context_length': 'context_length',
        'max_seq_len': 'context_length',
        'rope_theta': 'theta',
        'theta': 'theta'
    }
    
    # 构建模型参数
    model_kwargs = {
        'vocab_size': config.get('vocab_size', 18017),
        'context_length': config.get('context_length', config.get('max_seq_len', 256)),
        'd_model': config.get('d_model', 512),
        'n_layers': config.get('n_layers', config.get('num_layers', 4)),
        'n_heads': config.get('n_heads', config.get('num_heads', 16)),
        'd_ff': config.get('d_ff', 1344),
        'theta': config.get('theta', config.get('rope_theta', 10000.0)),
        # 消融实验参数
        'use_rmsnorm': config.get('use_rmsnorm', True),
        'norm_position': config.get('norm_position', 'pre'),
        'use_rope': config.get('use_rope', True),
        'use_swiglu': config.get('use_swiglu', True),
        'device': device
    }
    
    print("正在加载模型，配置信息：")
    for key, value in model_kwargs.items():
        if key != 'device':
            print(f"  {key}: {value}")
    
    # 创建模型
    model = TransformerLM(**model_kwargs).to(device)
    
    # 加载模型权重
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 假设整个checkpoint就是state_dict
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='使用训练好的Transformer语言模型生成文本')
    
    # 模型和分词器路径
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--vocab', type=str, required=True, help='分词器词汇表文件路径')
    parser.add_argument('--merges', type=str, required=True, help='分词器合并规则文件路径')
    
    # 生成参数
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='用于生成的输入提示')
    parser.add_argument('--max_tokens', type=int, default=256, help='要生成的最大token数量')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度（越低越确定）')
    parser.add_argument('--top_p', type=float, default=0.9, help='核采样的top-p阈值')
    parser.add_argument('--num_samples', type=int, default=1, help='要生成的样本数量')
    
    # 设备
    parser.add_argument('--device', type=str, default='auto', help='设备: auto, cpu, cuda, mps')
    
    # 序列结束标记
    parser.add_argument('--eos_token', type=str, default='<|endoftext|>', help='序列结束标记')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            merges_path=args.merges,
            device=device
        )
        print("模型和分词器加载成功！")
    except Exception as e:
        print(f"加载模型或分词器时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if tokenizer is None:
        print("错误: 无法加载分词器。请检查您的分词器实现。")
        return
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成文本
    print(f"\n正在生成 {args.num_samples} 个样本，提示词: '{args.prompt}'")
    print(f"参数: max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p}")
    print("-" * 80)
    
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n样本 {i+1}:")
            print("-" * 40)
        
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                eos_token=args.eos_token
            )
            
            print(generated_text)
            print("\n" + "=" * 80)
            
        except Exception as e:
            print(f"生成过程中出错: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == '__main__':
    main()