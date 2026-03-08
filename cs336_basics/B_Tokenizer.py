import regex
from collections import defaultdict
from typing import Iterable, Iterator, List, Set, Tuple, Dict
import torch
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self , vocab: Dict[int , bytes] , merges: List[Tuple[bytes , bytes]] , special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merge_priorities: Dict[Tuple[bytes , bytes], int] = {pair: i for i , pair in enumerate(self.merges)} # O(1) 查找合并优先级
        self.byte_to_id: Dict[bytes , int] = {v : k for k , v in self.vocab.items()} # O(1) 查找字节对应的token id    
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        从序列化文件加载分词器（tokenizer）的类方法。
        参数：vocab_filepath：指向 pickle 文件的路径，该文件包含一个字典，格式为 dict[int, bytes]（或类字节类型）。
             merges_filepath：指向 pickle 文件的路径，该文件包含一个列表，格式为 list[tuple[bytes, bytes]]（或类字符串类型）。
             special_tokens：可选参数，待注册/追加到词汇表中的字符串列表（list[str]）。
        返回：一个初始化完成的 Tokenizer 实例。
        """
        import pickle

        # 加载并归一化词汇表：键转为整数，值转为字节
        with open(vocab_filepath, "rb") as vf:
            raw_vocab = pickle.load(vf)

        norm_vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            kid = int(k)
            if isinstance(v, str):
                v = v.encode("utf-8")
            norm_vocab[kid] = v

        # 加载并标准化合并规则：确保为字节类型的元组
        with open(merges_filepath, "rb") as mf:
            raw_merges = pickle.load(mf)

        norm_merges: list[tuple[bytes, bytes]] = []
        for a, b in raw_merges:
            if isinstance(a, str):
                a = a.encode("utf-8")
            if isinstance(b, str):
                b = b.encode("utf-8")
            norm_merges.append((a, b))

        return cls(norm_vocab, norm_merges, special_tokens)
    
    # 应用BPE合并规则，将输入字节序列转换为token字节列表
    def _apply_bpe_merge(self , bytes_seq: bytes) -> List[bytes]:
        # 将输入字节序列分割成单字节的列表
        cur_bytes: List[bytes] = [bytes([b]) for b in bytes_seq]
        
        # 迭代指令合并
        while len(cur_bytes) > 1:
            # 1.找出当前序列中所有存在于合并规则中的token对
            exist_pairs: Set[Tuple[bytes , bytes]] = set()
            for i in range(len(cur_bytes) - 1):
                pair = (cur_bytes[i] , cur_bytes[i+1])
                if pair in self.merge_priorities:
                    exist_pairs.add(pair)
            if not exist_pairs:
                break
            
            # 2. 从存在token对中选择优先级最高（序号最小）的合并
            best_pair = min(exist_pairs , key=lambda pair : self.merge_priorities[pair])
            
            # 3. 执行合并
            new_bytes: List[bytes] = []
            i = 0
            while i < len(cur_bytes):
                if i < len(cur_bytes) - 1 and (cur_bytes[i] , cur_bytes[i+1]) == best_pair:
                    new_bytes.append(cur_bytes[i] + cur_bytes[i+1]) # 合并成新token
                    i += 2
                else:
                    new_bytes.append(cur_bytes[i])
                    i += 1  
                    
            # 4. 更新当前序列
            cur_bytes = new_bytes
            
        return cur_bytes
    
    # 将输入文本编码为token id列表
    def encode(self , text: str) -> List[int]:
        if not text:
            return []
        
        # 1. 处理特殊token
        sorted_special_tokens = sorted(self.special_tokens , key=len , reverse=True)
        special_token_pattern = '|'.join(map(regex.escape , sorted_special_tokens))
        
        # 2. 分割文本
        if self.special_tokens:
            chunks = regex.split(f'({special_token_pattern})' , text)
        else:
            chunks = [text]
        
        # 3. 处理分割后的文本
        token_ids: List[int] = []
        for chunk in chunks:
            # 1. 跳过空chunk
            if not chunk:
                continue
            
            # 2. 如果chunk是特殊token，直接编码为一个token id
            if chunk in self.special_tokens:
                special_token_bytes = chunk.encode("utf-8")
                special_token_id = self.byte_to_id[special_token_bytes]
                token_ids.append(special_token_id)
            # 3. 否则，对chunk进行预分词，并对每个token应用BPE合并，最后编码为token id
            else:
                for token in regex.findall(PAT , chunk):
                    bpe_token_bytes = self._apply_bpe_merge(token.encode("utf-8"))
                    
                    for token_bytes in bpe_token_bytes:
                        token_id = self.byte_to_id[token_bytes]
                        token_ids.append(token_id)
        
        # 4. 返回最终的token id列表
        return token_ids
    
    # 对输入文本列表进行编码，返回一个生成器，逐个生成文本对应的token id列表
    def encode_iterable(self , texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text) 
    
    # 将输入token id列表解码为文本
    def decode(self , token_ids: List[int]) -> str:
        bytes_seq = b''.join(self.vocab[id] for id in token_ids)
        return bytes_seq.decode("utf-8" , errors="replace")