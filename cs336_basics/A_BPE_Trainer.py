import os
from typing import List, Tuple, Dict, Set
import regex    
from collections import defaultdict
import pickle

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str],) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    给定一个输入语料库的路径，训练一个 BPE分词器，并输出其词汇表和合并规则。
    参数：
        input_path (str | os.PathLike): 指向 BPE 分词器训练数据的路径。
        vocab_size (int): 分词器词汇表中条目的总数（包括特殊标记）。
        special_tokens (list[str]): 一个字符串特殊标记列表，需要添加到分词器的词汇表中。
        这些字符串永远不会被拆分成多个标记，并且始终被视为单个标记。如果这些特殊标记出现在 input_path 中，它们会被当作普通字符串处理。
    返回值：
        vocab (dict[int, bytes]): 训练好的分词器的词汇表，这是一个从整数（词汇表中的标记 ID）到字节串（标记的字节表示）的映射。
        merges (list[tuple[bytes, bytes]]): BPE 合并规则。列表中的每一项都是一个字节串元组 (<token1>, <token2>)，表示 <token1> 与 <token2> 进行了合并。合并规则按照创建顺序排列。
    """
    # 1.创建基础词汇表
    vocab: Dict[int , bytes] = {i : bytes([i]) for i in range(256)} #初始化词汇表
    cur_id: int = 256 # 新token的id
    cur_bytes: Set[bytes] = set(vocab.values()) #快速查找字节值
    
    # 2.添加特殊token
    for spe_t in special_tokens:
        if len(vocab) >= vocab_size:
            break
        spe_b = spe_t.encode("utf-8") # 将特殊token转换为字节序列
        if spe_b not in cur_bytes:
            vocab[cur_id] = spe_b
            cur_bytes.add(spe_b)
            cur_id += 1
    
    # 3.加载语料库
    try:
        with open(input_path , "r" , encoding="utf-8" , errors="ignore") as file:
            text = file.read()
    except FileNotFoundError:
        text = " "
    
    # 4.预分词
    # escaped_tokens = [regex.escape(spe_t) for spe_t in special_tokens] # 预处理：转义所有特殊token
    # pattern = "|".join(escaped_tokens) # 构建正则表达式模式
    # text_chunks = regex.split(pattern, text) # 执行分割
    text_chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)
    # 定义GPT-2风格的预分词正则表达式，用于将文本分割成更易于处理的“子词单元”。该模式会尽量保留单词、数字和标点符号的完整性。
    PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # 统计初始token频率
    token_freqs = defaultdict(int)
    for chunk in text_chunks:
        for word in regex.findall(PRETOKENIZER_PATTERN, chunk):
            word_bytes = word.encode("utf-8")
            bytes_list = [bytes([b]) for b in word_bytes]
            token_freqs[tuple(bytes_list)] += 1
    
    # 5.迭代合并
    merges: List[Tuple[bytes , bytes]] = [] # 合并记录
    pair_freqs = defaultdict(int) # 相邻token对频率表
    for token_seq , freq in token_freqs.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i] , token_seq[i+1])
            pair_freqs[pair] += freq
    # 优化
    token_seq_to_pairs = {} # 预处理每个token序列包含的token对
    for token_seq in token_freqs.keys():
        pairs_in_seq = set()
        for i in range(len(token_seq) - 1):
            pairs_in_seq.add((token_seq[i], token_seq[i+1]))
        token_seq_to_pairs[token_seq] = pairs_in_seq
            
    while len(vocab) < vocab_size:
        if not pair_freqs:
            break
        
        # 选择频率最高的合并对
        max_freq = max(pair_freqs.values())
        most_freq_pairs = [pair for pair , freq in pair_freqs.items() if freq == max_freq]
        best_pair = max(most_freq_pairs)
        
        # 创建新token并更新词汇表
        new_token = best_pair[0] + best_pair[1]
        vocab[cur_id] = new_token
        merges.append(best_pair)
        cur_id += 1
        
        # 本次合并影响的token序列
        affected_tokens_freqs = [
            (token_seq, freq)
            for token_seq, freq in token_freqs.items() 
            if best_pair in token_seq_to_pairs[token_seq] # O(1) 查找
        ]
        # affected_tokens_freqs = [
        #     (token_seq, freq) # 保存序列及其频率
        #     for token_seq, freq in token_freqs.items() 
        #     if any(token_seq[i:i+2] == best_pair for i in range(len(token_seq) - 1)) # 检查序列中是否存在该对
        # ]
        
        for token_seq , freq in affected_tokens_freqs:
            # 从相邻token对频率表中移除旧token序列的贡献
            for i in range(len(token_seq) - 1):
                old_pair = (token_seq[i] , token_seq[i+1])
                pair_freqs[old_pair] -= freq
                if pair_freqs[old_pair] <= 0:
                    del pair_freqs[old_pair]
            
            # 构建新token序列（将所有出现的best_pair合并为new_token）
            new_token_seq = []
            i = 0
            while i < len(token_seq):
                if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == best_pair:
                    new_token_seq.append(new_token) # 新token
                    i += 2
                else:
                    new_token_seq.append(token_seq[i])
                    i += 1
            new_token_seq = tuple(new_token_seq)
            
            # 将新token序列产生的对频率添加到pair_freqs中
            for i in range(len(new_token_seq) - 1):
                new_pair = (new_token_seq[i], new_token_seq[i+1])
                pair_freqs[new_pair] += freq
                
            # 更新token频率表和token序列到token对的映射
            del token_freqs[token_seq]
            token_freqs[new_token_seq] = freq
            del token_seq_to_pairs[token_seq]
            new_pairs_set = set()
            for i in range(len(new_token_seq) - 1):
                new_pairs_set.add((new_token_seq[i], new_token_seq[i+1]))
            token_seq_to_pairs[new_token_seq] = new_pairs_set

            
    with open("./data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("./data/merges.pkl", "wb") as f:
        pickle.dump(merges, f)
        
    return vocab, merges

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-valid.txt", 
        20000, 
        special_tokens
    )
    print(f"训练完成，最终词汇表大小: {len(vocab)}")  # 18017
    print(f"生成了 {len(merges)} 次合并。")  # 17760
    # print(f"具体词汇表和合并规则如下：")
    # print(vocab) # 输出量较大
    # print(merges) # 输出量较大