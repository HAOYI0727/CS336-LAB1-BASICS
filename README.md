# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
---

# CS336-Assignment1-Basics极简使用说明

## 目录说明

- checkpoints：基准实验经过5000steps训练得到的模型信息
- cs336_basics：实验任务书中全部实验代码（顺序排列）
- data：实验训练与验证数据、分词器信息与文本转换
  - 分词器信息：vocab.pkl、merges.pkl
  - 训练文本：TinyStoriesV2-GPT4-valid.txt -> train.dat
  - 验证文本：TinyStoriesV2-GPT4-LittleTest.txt -> valid.dat
  - >注意：.txt文件需要转换成.dat文件，运行data/Prepare_data.py即可（`uv run python data/Prepare_data.py`）
- tests：测试文件，终端运行`uv run pytest`即可验证（测试全部通过）
  
## 使用说明

- BPE分词器：运行cs336_basics/A_BPE_Trainer.py代码即可
  - `uv run python cs336_basics/A_BPE_Trainer.py`
- 训练与消融实验：运行cs336_basics/U_Trainer.py代码+添加必要参数
  - `uv run python cs336_basics/U_Trainer.py --data_dir ./data --is_base_experiment --wandb_run_name "01_base_model" --vocab_size 18017`
  - `uv run python /root/CS336_lab1/cs336_basics/U_Trainer.py --data_dir ./data --no_rmsnorm --wandb_run_name "02_ablation_no_rmsnorm" --vocab_size 18017`
  - `uv run python /root/CS336_lab1/cs336_basics/U_Trainer.py --data_dir ./data --norm_position post --wandb_run_name "03_ablation_post_norm" --vocab_size 18017`
  - `uv run python /root/CS336_lab1/cs336_basics/U_Trainer.py  --data_dir ./data --no_rope --wandb_run_name "04_ablation_no_rope" --vocab_size 18017`
  - `uv run python /root/CS336_lab1/cs336_basics/U_Trainer.py --data_dir ./data --use_silu --wandb_run_name "05_ablation_silu" --vocab_size 18017`
- 生成文本器：运行cs336_basics/T_Generate_text.py代码+添加必要参数
  - `uv run python cs336_basics/T_Generate_text.py --checkpoint checkpoints/base_experiment/checkpoint_final_4999.pt --vocab data/vocab.pkl --merges data/merges.pkl`
