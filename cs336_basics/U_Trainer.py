"""
Transformer语言模型训练脚本 - 用于消融实验
"""
import argparse
import os
import sys
import json
import torch
import pathlib
import numpy as np
import time
from tqdm import tqdm
import wandb

# 导入我们的模块
from cs336_basics.M_Transformer_LM import TransformerLM
from cs336_basics.O_AdamW import AdamW
from cs336_basics.R_Dataloader import data_loading
from cs336_basics.N_Cross_entropy_loss import cross_entropy_loss
from cs336_basics.P_Learning_rate_cosine_schedule import cosine_schedule
from cs336_basics.Q_Gradient_clipping import gradient_clipping
from cs336_basics.S_Checkpoint import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='训练Transformer语言模型')
    
    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=18017, help='词汇表大小')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--d_ff', type=int, default=1344, help='前馈网络维度')
    parser.add_argument('--context_len', type=int, default=256, help='最大序列长度')
    parser.add_argument('--num_heads', type=int, default=16, help='注意力头数量')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta参数')
    
    # 消融实验参数
    parser.add_argument('--use_rmsnorm', action='store_true', default=True, help='使用RMSNorm')
    parser.add_argument('--no_rmsnorm', action='store_false', dest='use_rmsnorm', help='禁用RMSNorm')
    parser.add_argument('--norm_position', type=str, choices=['pre', 'post'], default='pre', 
                        help='归一化位置: pre 或 post')
    parser.add_argument('--use_rope', action='store_true', default=True, help='使用RoPE位置编码')
    parser.add_argument('--no_rope', action='store_false', dest='use_rope', help='禁用RoPE')
    parser.add_argument('--use_swiglu', action='store_true', default=True, help='使用SwiGLU激活函数')
    parser.add_argument('--use_silu', action='store_false', dest='use_swiglu', help='使用FFNSiLU激活函数')
    
    # 优化器参数
    parser.add_argument('--max_lr', type=float, default=1e-3, help='最大学习率')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='最小学习率')
    parser.add_argument('--warm_up_it', type=int, default=500, help='预热迭代次数')
    parser.add_argument('--cosine_it', type=int, default=10000, help='余弦退火迭代次数')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='权重衰减')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='梯度裁剪范数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--train_steps', type=int, default=5000, help='总训练步数')
    parser.add_argument('--val_interval', type=int, default=100, help='验证间隔')
    parser.add_argument('--val_batches', type=int, default=10, help='验证批次数量')
    parser.add_argument('--save_intervals', type=int, default=1000, help='检查点保存间隔（仅基础模型）')
    parser.add_argument('--log_intervals', type=int, default=1, help='日志记录间隔')
    parser.add_argument('--save_ckp_path', type=str, default='./checkpoints', help='检查点保存目录')
    parser.add_argument('--resume_ckp', type=str, default=None, help='从中恢复的检查点路径')
    parser.add_argument('--experiment_name', type=str, default=None, help='用于保存检查点的实验名称')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--device', type=str, default='auto', help='设备: auto, cpu, cuda, mps')
    
    # Wandb参数
    parser.add_argument('--wandb_project', type=str, default='cs336-transformer-ablation', help='Wandb项目名称')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb运行名称')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Wandb标签')
    parser.add_argument('--no_wandb', action='store_true', help='禁用wandb日志记录')
    
    # 实验类型
    parser.add_argument('--is_base_experiment', action='store_true', help='这是基础实验（将保存检查点）')
    
    return parser.parse_args()

def get_device(device_arg):
    """获取合适的设备"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg

def get_dataset_memmap(path, dtype=np.uint16):
    """使用内存映射加载数据集以提高效率"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据文件: {path}")
    dataset = np.memmap(path, dtype=dtype, mode='r')
    return dataset

def compute_perplexity(logits, targets):
    """计算每个token的困惑度"""
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.exp(-target_log_probs.mean())

def compute_gradient_norm(model):
    """计算梯度范数以供监控"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def validate_model(model, val_data, args, device):
    """简化的验证 - 返回损失和困惑度"""
    model.eval()
    val_losses = []
    val_ppls = []
    
    with torch.no_grad():
        for _ in range(args.val_batches):
            val_input_ids, val_target_ids = data_loading(
                val_data,
                args.batch_size,
                args.context_len,
                device
            )
            
            val_input_ids = val_input_ids.long().to(device)
            val_target_ids = val_target_ids.long().to(device)
            
            val_logits = model(val_input_ids)
            val_logits_flat = val_logits.view(-1, val_logits.size(-1))
            val_targets_flat = val_target_ids.view(-1)
            
            val_loss = cross_entropy_loss(val_logits_flat, val_targets_flat)
            val_ppl = compute_perplexity(val_logits_flat, val_targets_flat)
            
            val_losses.append(val_loss.item())
            val_ppls.append(val_ppl.item())
    
    model.train()
    return {
        'loss': float(np.mean(val_losses)),
        'perplexity': float(np.mean(val_ppls))
    }

def get_experiment_name(args):
    """根据消融配置生成实验名称"""
    if args.experiment_name:
        return args.experiment_name
    
    if args.is_base_experiment:
        return "base_experiment"
    
    norm_str = "rmsnorm" if args.use_rmsnorm else "no_rmsnorm"
    pos_str = f"{args.norm_position}_norm"
    rope_str = "rope" if args.use_rope else "nope"
    ffn_str = "swiglu" if args.use_swiglu else "silu"
    return f"{norm_str}-{pos_str}-{rope_str}-{ffn_str}"

def main():
    args = parse_args()
    
    # 如果需要，调整SiLU的d_ff
    if not args.use_swiglu:
        expected_d_ff = 4 * args.d_model
        if args.d_ff != expected_d_ff:
            print(f"注意: 使用FFNSiLU，d_ff={args.d_ff}（标准应为4*d_model={expected_d_ff}）")
    
    # 设置设备
    device = get_device(args.device)
    print(f"使用设备: {device}")
    
    # 生成实验名称
    experiment_name = get_experiment_name(args)
    
    # 创建实验特定的检查点目录（仅为基础实验）
    experiment_checkpoint_path = None
    if args.is_base_experiment:
        experiment_checkpoint_path = os.path.join(args.save_ckp_path, experiment_name)
        os.makedirs(experiment_checkpoint_path, exist_ok=True)
        print(f"基础实验 - 检查点将保存到: {experiment_checkpoint_path}")
    else:
        print(f"消融实验 - 不会保存检查点")
    
    # 为wandb创建标签
    tags = args.wandb_tags.copy()
    tags.extend([
        'base' if args.is_base_experiment else 'ablation',
        'rmsnorm' if args.use_rmsnorm else 'no_rmsnorm',
        f'{args.norm_position}_norm',
        'rope' if args.use_rope else 'no_rope',
        'swiglu' if args.use_swiglu else 'silu'
    ])
    
    # 初始化wandb
    if not args.no_wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = experiment_name
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            tags=tags
        )
        print(f"Wandb已初始化: {wandb.run.name}")
    
    # 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_len,
        d_model=args.d_model,
        n_layers=args.num_layers,
        n_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        use_rmsnorm=args.use_rmsnorm,
        norm_position=args.norm_position,
        use_rope=args.use_rope,
        use_swiglu=args.use_swiglu,
        device=device
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型初始化完成，参数量: {total_params:,}")
    print(f"实验: {experiment_name}")
    print(f"配置: RMSNorm={args.use_rmsnorm}, 归一化位置={args.norm_position}, "
          f"RoPE={args.use_rope}, FFN={'SwiGLU' if args.use_swiglu else 'FFNSiLU'}")
    print(f"实验类型: {'基础（将保存检查点）' if args.is_base_experiment else '消融（不保存检查点）'}")
    
    if not args.no_wandb:
        wandb.run.summary["total_parameters"] = total_params
        wandb.run.summary["experiment_name"] = experiment_name
        wandb.run.summary["is_base_experiment"] = args.is_base_experiment
    
    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # 加载数据集
    train_data_path = os.path.join(args.data_dir, 'train.dat')
    val_data_path = os.path.join(args.data_dir, 'valid.dat')
    
    train_data = get_dataset_memmap(train_data_path)
    val_data = get_dataset_memmap(val_data_path)
    
    print(f"训练数据: {len(train_data):,} 个token")
    print(f"验证数据: {len(val_data):,} 个token")
    
    # 从检查点恢复（仅为基础实验）
    start_iter = 0
    if args.resume_ckp and args.is_base_experiment:
        # 如果恢复路径不包含实验名称，假设它在实验目录中
        if not os.path.exists(args.resume_ckp):
            possible_path = os.path.join(experiment_checkpoint_path, args.resume_ckp)
            if os.path.exists(possible_path):
                args.resume_ckp = possible_path
        
        print(f"从检查点恢复: {args.resume_ckp}")
        start_iter = load_checkpoint(args.resume_ckp, model, optimizer)
        print(f"从迭代 {start_iter} 恢复")
    elif args.resume_ckp and not args.is_base_experiment:
        print("警告: 消融实验禁用了检查点恢复功能")
    
    # 保存实验配置（仅为基础实验）
    if args.is_base_experiment:
        config_path = os.path.join(experiment_checkpoint_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # 训练循环
    model.train()
    train_losses = []
    train_ppls = []
    best_val_loss = float('inf')
    
    # 创建进度条
    pbar = tqdm(range(start_iter, args.train_steps), 
                desc=f"训练 {experiment_name}", 
                initial=start_iter, 
                total=args.train_steps)
    
    for iter_num in pbar:
        # 获取当前迭代的学习率
        lr = cosine_schedule(
            iter_num,
            args.max_lr,
            args.min_lr,
            args.warm_up_it,
            args.cosine_it
        )
        
        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 采样批次
        input_ids, target_ids = data_loading(
            train_data,
            args.batch_size,
            args.context_len,
            device=device
        )
        
        # 移动到设备
        input_ids = input_ids.long().to(device)
        target_ids = target_ids.long().to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # 计算损失
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)
        
        loss = cross_entropy_loss(logits_flat, targets_flat)
        
        # 为监控计算困惑度
        with torch.no_grad():
            ppl = compute_perplexity(logits_flat, targets_flat)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪并计算梯度范数
        gradient_clipping(model.parameters(), args.clip_grad_norm)
        grad_norm = compute_gradient_norm(model)
        
        # 优化器步进
        optimizer.step()
        
        # 跟踪指标
        train_losses.append(loss.item())
        train_ppls.append(ppl.item())
        
        # 日志记录
        if iter_num % args.log_intervals == 0:
            avg_loss = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else np.mean(train_losses)
            avg_ppl = np.mean(train_ppls[-100:]) if len(train_ppls) >= 100 else np.mean(train_ppls)
            
            # 更新进度条
            pbar.set_postfix({
                '损失': f'{loss.item():.4f}',
                '困惑度': f'{ppl.item():.2f}',
                '平均损失': f'{avg_loss:.4f}',
                '学习率': f'{lr:.2e}'
            })
            
            # 记录到wandb
            if not args.no_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/avg_loss': avg_loss,
                    'train/perplexity': ppl.item(),
                    'train/avg_perplexity': avg_ppl,
                    'train/learning_rate': lr,
                    'train/gradient_norm': grad_norm,
                    'iteration': iter_num
                })
        
        # 验证
        if iter_num % args.val_interval == 0 and iter_num > 0:
            val_metrics = validate_model(model, val_data, args, device)
            
            # 打印验证结果
            tqdm.write(f"\n[{experiment_name}] 迭代 {iter_num}: 验证损失={val_metrics['loss']:.4f}, 验证困惑度={val_metrics['perplexity']:.2f}")
            
            # 记录到wandb
            if not args.no_wandb:
                wandb.log({
                    'val/loss': val_metrics['loss'],
                    'val/perplexity': val_metrics['perplexity'],
                    'iteration': iter_num
                })
                
                # 在wandb摘要中更新最佳验证损失
                if val_metrics['loss'] < best_val_loss:
                    wandb.run.summary["best_val_loss"] = val_metrics['loss']
                    wandb.run.summary["best_val_loss_iter"] = iter_num
                    best_val_loss = val_metrics['loss']
        
        # 仅为基础实验保存检查点
        # 定期保存检查点 - 每1000轮
        if args.is_base_experiment and iter_num % args.save_intervals == 0 and iter_num > 0:
            # 1. 保存定期检查点
            checkpoint_path = os.path.join(experiment_checkpoint_path, f'checkpoint_{iter_num}.pt')
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            tqdm.write(f"[{experiment_name}] 检查点已保存: {checkpoint_path}")
            
            # 2. 如果当前模型是最佳的，也保存为最佳模型
            if val_metrics['loss'] == best_val_loss:  # 或者当前验证损失是最佳的
                best_checkpoint_path = os.path.join(experiment_checkpoint_path, 'best_model.pt')
                save_checkpoint(model, optimizer, iter_num, best_checkpoint_path)
                tqdm.write(f"[{experiment_name}] 最佳模型已保存于迭代 {iter_num}！")
    
    # 关闭进度条
    pbar.close()
    
    # 保存最终检查点（仅为基础实验）
    if args.is_base_experiment:
        final_checkpoint_path = os.path.join(experiment_checkpoint_path, f'checkpoint_final_{iter_num}.pt')
        save_checkpoint(model, optimizer, iter_num, final_checkpoint_path)
        print(f"[{experiment_name}] 最终检查点已保存: {final_checkpoint_path}")
    
    # 将最终模型保存到wandb（仅为基础实验）
    if not args.no_wandb and args.is_base_experiment:
        # 使用实验名称创建构件
        artifact = wandb.Artifact(
            name=f"model-{experiment_name}-{wandb.run.id}",
            type="model",
            metadata=dict(args.__dict__)
        )
        artifact.add_file(final_checkpoint_path)
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)
        
        print(f"最终模型已保存到wandb构件")
    
    # 所有实验的最终wandb日志记录
    if not args.no_wandb:
        # 最终摘要
        wandb.run.summary["final_iteration"] = iter_num
        wandb.run.summary["final_train_loss"] = np.mean(train_losses[-100:])
        wandb.run.summary["final_train_perplexity"] = np.mean(train_ppls[-100:])
        
        print(f"实验数据已记录到wandb")
        wandb.finish()
    
    # 创建实验摘要文件（仅为基础实验）
    if args.is_base_experiment:
        summary_path = os.path.join(experiment_checkpoint_path, 'summary.json')
        summary = {
            'experiment_name': experiment_name,
            'final_iteration': iter_num,
            'final_train_loss': float(np.mean(train_losses[-100:])),
            'final_train_perplexity': float(np.mean(train_ppls[-100:])),
            'best_val_loss': float(best_val_loss),
            'config': vars(args)
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"实验摘要已保存: {summary_path}")
    
    print(f"实验 {experiment_name} 完成！")

if __name__ == '__main__':
    main()