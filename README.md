# CS336-LAB1-BASICS

该仓库是斯坦福大学2025年春季CS336课程（Language Models from Scratch）的第一次作业实现，聚焦大语言模型基础开发，包含**BPE分词器实现**、**Transformer基础模型训练**、**核心模块消融实验及文本生成**全流程代码，基于TinyStories小样本数据集完成实验，采用`uv`进行环境管理保证复现性。

## 课程作业背景

CS336课程作业1要求从零实现大模型基础组件（仅使用PyTorch原语，禁止直接调用高层API），核心任务包括：
1. 实现BPE（Byte Pair Encoding）分词器，解决文本向量化的OOV问题；
2. 构建基础Transformer架构，实现Adam优化器；
3. 在小样本数据集上完成模型训练，并对RMSNorm、RoPE、激活函数等核心模块做消融实验；
4. 实现训练后模型的文本生成功能。

作业详细要求见仓库内`cs336_spring2025_assignment1_basics.pdf`。

## 仓库目录结构
```
CS336-LAB1-BASICS/
├── checkpoints/        # 基准实验5000步训练的模型权重文件（自行训练）
├── cs336_basics/       # 核心实验代码（按任务顺序A~U排列）
├── data/               # 训练/验证数据、分词器文件、数据预处理代码
├── figures/            # 消融实验结果可视化图表
├── tests/              # 单元测试文件，验证代码实现正确性
├── CHANGELOG.md        # 版本更新日志
├── LICENSE             # 开源协议
├── README.md           # 项目说明文档
├── cs336_spring2025_assignment1_basics.pdf  # 作业任务书
├── make_submission.sh  # 作业提交脚本
├── pyproject.toml      # uv环境配置文件
├── uv.lock             # 环境依赖锁定文件
```

###### 关键目录说明

- **cs336_basics**：包含BPE分词器训练、模型组件的组合与实现、模型训练与验证、消融实验、测试文本生成的全部可执行代码；
- **data**：存储原始文本数据、预处理后的`.dat`数据、分词器词表（`vocab.pkl`）和合并规则（`merges.pkl`）；
- **tests**：通过单元测试验证代码实现，已完成`adapters.py`中函数对接自定义实现，可通过终端运行`uv run pytest`一键测试；
- **checkpoints**：基准模型训练完成的权重，可直接用于文本生成（由于模型checkpoints过大，由读者自行训练实现。

## 环境搭建

本项目使用**uv**进行环境管理，保证跨平台的可复现性、便捷性，替代传统`conda`/`pip`环境管理方式。

###### 1. 安装uv
```bash
# 方式1：官方推荐安装（跨平台）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 方式2：pip安装
pip install uv
# 方式3：brew安装（Mac/Linux）
brew install uv
```

###### 2. 运行项目代码
uv会自动解析`pyproject.toml`和`uv.lock`，自动创建并激活环境，运行代码命令格式：
```bash
uv run <python_file_path>
```

## 数据准备
实验仅采用**TinyStoriesV2-GPT4**数据集，需先下载原始数据并完成预处理。

###### 1. 下载原始数据

```bash
# 创建data目录并进入
mkdir -p data && cd data
# 下载TinyStories数据集
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
# 返回根目录
cd ..
```

###### 2. 数据预处理
将原始`.txt`文本转换为模型可读取的`.dat`格式，运行预处理脚本：
```bash
uv run python data/Prepare_data.py
```
预处理后生成：
- 训练数据：`train.dat`（来自TinyStoriesV2-GPT4-valid.txt）
- 验证数据：`valid.dat`（来自TinyStoriesV2-GPT4-LittleTest.txt）

## 快速开始

###### 1. 单元测试验证

初始化时测试会因`NotImplementedError`失败，由于笔者已完成`./tests/adapters.py`中函数实现后，可直接运行测试验证代码正确性：
```bash
uv run pytest
```
全部测试通过表示基础代码实现无误。

###### 2. 训练BPE分词器
实现基于字节对编码的分词器，生成词表`vocab.pkl`和合并规则`merges.pkl`（存储至`data/`目录）：
```bash
uv run python cs336_basics/A_BPE_Trainer.py
```
BPE分词器是GPT系列模型的核心分词方案，通过合并高频字符对生成子词表，平衡OOV问题和语义保留能力。

###### 3. 模型训练与消融实验
通过`U_Trainer.py`完成**基准模型训练**和**核心模块消融实验**，实验围绕大模型关键组件展开，对比不同模块对模型性能的影响。

###### 3.1. 基准模型训练（含RMSNorm、Pre-Norm、RoPE、SwiGLU）

```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --is_base_experiment --wandb_run_name "01_base_model" --vocab_size 18017
```

###### 3.2. 消融实验（逐一移除/替换核心模块）

1. 移除RMSNorm（对比no_RMSNorm，验证归一化方案影响）
```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --no_rmsnorm --wandb_run_name "02_ablation_no_rmsnorm" --vocab_size 18017
```

2. 后置归一化（替换默认前置归一化，验证归一化位置影响）
```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --norm_position post --wandb_run_name "03_ablation_post_norm" --vocab_size 18017
```

3. 移除RoPE（旋转位置编码，验证位置编码对模型的影响）
```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --no_rope --wandb_run_name "04_ablation_no_rope" --vocab_size 18017
```

4. 使用SiLU激活函数（替换SwiGLU，验证激活函数性能）
```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --use_silu --wandb_run_name "05_ablation_silu" --vocab_size 18017
```

###### 4. 文本生成
使用训练完成的基准模型权重（或自定义训练权重）实现文本生成，需指定**模型检查点**、**分词器词表**和**合并规则**：
```bash
uv run python cs336_basics/T_Generate_text.py --checkpoint checkpoints/base_experiment/checkpoint_final_5000.pt --vocab data/vocab.pkl --merges data/merges.pkl
```
可替换`--checkpoint`为自定义消融实验的模型权重，对比不同模型的生成效果。

## 核心技术说明

本实验涉及大模型基础核心技术，关键模块说明如下：
1. **RMSNorm**：简化的LayerNorm，移除均值减法步骤，降低计算量和内存占用，是现代LLM（如Llama、Falcon）的标准归一化方案；
2. **RoPE（旋转位置编码）**：赋予模型位置感知能力，提升长序列外推性能，是大模型位置编码的主流方案；
3. **BPE分词**：解决传统词级分词的OOV问题，通过子词合并平衡分词粒度和语义保留；
4. **SiLU激活函数**：光滑的自门控激活函数，相比ReLU/GELU具有更优的梯度特性，是SwiGLU的核心组件。

---

# CS336-LAB1-BASICS

This repository implements the first assignment of Stanford University's Spring 2025 CS336 course (Language Models from Scratch), focusing on foundational development of large language models (LLMs). It includes complete code for **BPE tokenizer implementation**, **basic Transformer model training**, **core module ablation experiments**, and **text generation**—all conducted on the TinyStories small-scale dataset with `uv` for environment management to ensure reproducibility.

## Course Assignment Background

Assignment 1 of CS336 requires implementing foundational LLM components from scratch (using only PyTorch primitives, no direct calls to high-level APIs). Core tasks include:
1. Implement a **Byte Pair Encoding (BPE)** tokenizer to address the out-of-vocabulary (OOV) problem in text vectorization;
2. Build a basic Transformer architecture and implement the Adam optimizer;
3. Train the model on a small-scale dataset and conduct ablation experiments on core modules (RMSNorm, RoPE, activation functions, etc.);
4. Implement text generation functionality with the trained model.

Detailed assignment requirements are available in `cs336_spring2025_assignment1_basics.pdf` in the repository.

## Repository Directory Structure
```
CS336-LAB1-BASICS/
├── checkpoints/        # Model weight files from 5000-step baseline experiment (train your own)
├── cs336_basics/       # Core experiment code (organized by task sequence A~U)
├── data/               # Training/validation data, tokenizer files, data preprocessing code
├── figures/            # Visualization charts for ablation experiment results
├── tests/              # Unit test files to verify code correctness
├── CHANGELOG.md        # Version update log
├── LICENSE             # Open source license
├── README.md           # Project documentation
├── cs336_spring2025_assignment1_basics.pdf  # Assignment specification
├── make_submission.sh  # Assignment submission script
├── pyproject.toml      # uv environment configuration file
├── uv.lock             # Environment dependency lock file
```

### Key Directory Explanations

- **cs336_basics**: Contains all executable code for BPE tokenizer training, model component assembly/implementation, model training/validation, ablation experiments, and test text generation;
- **data**: Stores raw text data, preprocessed `.dat` data, tokenizer vocabulary (`vocab.pkl`), and merge rules (`merges.pkl`);
- **tests**: Validates code implementation through unit tests. Functions in `adapters.py` have been customized to interface with our implementations—run tests with a single terminal command;
- **checkpoints**: Weights from baseline model training (directly usable for text generation). Due to large file size, readers need to train models independently.

## Environment Setup

This project uses **uv** for environment management (replacing traditional `conda`/`pip`) to ensure cross-platform reproducibility and ease of use.

### 1. Install uv
```bash
# Method 1: Official recommended installation (cross-platform)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Method 2: Install via pip
pip install uv
# Method 3: Install via brew (Mac/Linux)
brew install uv
```

### 2. Run Project Code
uv automatically parses `pyproject.toml` and `uv.lock` to create/activate the environment. Use this command format to run code:
```bash
uv run <python_file_path>
```

## Data Preparation

Experiments use only the **TinyStoriesV2-GPT4** dataset. Download raw data and complete preprocessing first.

### 1. Download Raw Data

```bash
# Create and enter data directory
mkdir -p data && cd data
# Download TinyStories dataset
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
# Return to root directory
cd ..
```

### 2. Data Preprocessing
Convert raw `.txt` text to model-readable `.dat` format by running the preprocessing script:
```bash
uv run python data/Prepare_data.py
```
After preprocessing, the following files are generated:
- Training data: `train.dat` (from TinyStoriesV2-GPT4-valid.txt)
- Validation data: `valid.dat` (from TinyStoriesV2-GPT4-LittleTest.txt)

## Quick Start

### 1. Unit Test Validation

Initial tests will fail with `NotImplementedError`. After implementing functions in `./tests/adapters.py`, run tests to verify code correctness:
```bash
uv run pytest
```
All passing tests confirm correct implementation of basic code.

### 2. Train BPE Tokenizer
Implement a BPE-based tokenizer to generate vocabulary (`vocab.pkl`) and merge rules (`merges.pkl`) (stored in `data/` directory):
```bash
uv run python cs336_basics/A_BPE_Trainer.py
```
BPE tokenization is the core tokenization scheme for GPT-series models. It generates subword vocabularies by merging frequent character pairs, balancing OOV handling and semantic preservation.

### 3. Model Training & Ablation Experiments
Complete **baseline model training** and **core module ablation experiments** via `U_Trainer.py`. Experiments focus on key LLM components to compare their impact on model performance.

#### 3.1. Baseline Model Training (with RMSNorm, Pre-Norm, RoPE, SwiGLU)

```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --is_base_experiment --wandb_run_name "01_base_model" --vocab_size 18017
```

#### 3.2. Ablation Experiments (remove/replace core modules one by one)

1. Remove RMSNorm (compare with no_RMSNorm to verify impact of normalization scheme)
```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --no_rmsnorm --wandb_run_name "02_ablation_no_rmsnorm" --vocab_size 18017
```

2. Post-Normalization (replace default pre-normalization to verify impact of normalization position)
```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --norm_position post --wandb_run_name "03_ablation_post_norm" --vocab_size 18017
```

3. Remove RoPE (Rotary Position Embedding to verify impact of positional encoding on model)
```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --no_rope --wandb_run_name "04_ablation_no_rope" --vocab_size 18017
```

4. Use SiLU activation function (replace SwiGLU to verify activation function performance)
```bash
uv run python cs336_basics/U_Trainer.py --data_dir ./data --use_silu --wandb_run_name "05_ablation_silu" --vocab_size 18017
```

### 4. Text Generation
Generate text using trained baseline model weights (or custom-trained weights). Specify the **model checkpoint**, **tokenizer vocabulary**, and **merge rules**:
```bash
uv run python cs336_basics/T_Generate_text.py --checkpoint checkpoints/base_experiment/checkpoint_final_5000.pt --vocab data/vocab.pkl --merges data/merges.pkl
```
Replace `--checkpoint` with weights from custom ablation experiments to compare generation performance across models.

## Core Technology Explanations

This experiment involves foundational core technologies for LLMs. Key module explanations:
1. **RMSNorm**: Simplified LayerNorm that removes mean subtraction, reducing computation and memory usage—now the standard normalization scheme for modern LLMs (e.g., Llama, Falcon);
2. **RoPE (Rotary Position Embedding)**: Provides positional awareness to models, improving long-sequence extrapolation performance (mainstream positional encoding for LLMs);
3. **BPE Tokenization**: Addresses OOV issues in traditional word-level tokenization, balancing token granularity and semantic preservation via subword merging;
4. **SiLU Activation Function**: Smooth self-gated activation function with better gradient properties than ReLU/GELU—core component of SwiGLU.

### Summary
1. This repository implements foundational LLM components (BPE tokenizer, Transformer, optimizer) from scratch using PyTorch primitives, following Stanford CS336 assignment requirements;
2. The workflow includes environment setup with `uv`, data preparation (TinyStories dataset), model training, ablation experiments on core modules (RMSNorm/RoPE/activation functions), and text generation;
3. Key technical choices (RMSNorm, RoPE, BPE) align with modern LLM design principles, with ablation experiments validating their impact on model performance.
