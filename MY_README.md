### CS336-Assignment1-Basics极简使用说明

#### 目录说明

- checkpoints：基准实验经过5000steps训练得到的模型信息
- cs336_basics：实验任务书中全部实验代码（顺序排列）
- data：实验训练与验证数据、分词器信息与文本转换
  - 分词器信息：vocab.pkl、merges.pkl
  - 训练文本：TinyStoriesV2-GPT4-valid.txt -> train.dat
  - 验证文本：TinyStoriesV2-GPT4-LittleTest.txt -> valid.dat
  - >注意：.txt文件需要转换成.dat文件，运行data/Prepare_data.py即可（`uv run python data/Prepare_data.py`）
- tests：测试文件，终端运行`uv run pytest`即可验证（测试全部通过）
  
#### 使用说明

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