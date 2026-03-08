import numpy as np
from cs336_basics.B_Tokenizer import Tokenizer

def text_to_dat(text_file, dat_file, tokenizer):
    """
    将文本文件转换为token ID的.dat文件
    """
    # 读取文本
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # tokenize（将文本转换为整数ID）
    tokens = tokenizer.encode(text)
    
    # 转换为numpy数组
    token_array = np.array(tokens, dtype=np.uint16)
    
    # 保存为.dat文件
    fp = np.memmap(dat_file, dtype=np.uint16, mode='w+', shape=token_array.shape)
    fp[:] = token_array[:]
    fp.flush()
    
    print(f"Converted {text_file} -> {dat_file}")
    print(f"  {len(tokens)} tokens")

tokenizer = Tokenizer.from_files('data/vocab.pkl', 'data/merges.pkl')
text_to_dat('data/TinyStoriesV2-GPT4-valid.txt', 'data/train.dat', tokenizer)
text_to_dat('data/TinyStoriesV2-GPT4-LittleTest.txt', 'data/valid.dat', tokenizer)