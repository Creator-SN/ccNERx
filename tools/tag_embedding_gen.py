from typing import Dict, List
from text2vec import Word2Vec
import os

class EmbeddingGenerator():
    def __init__(self) -> None:
        self.model = Word2Vec('w2v-light-tencent-chinese')
        self.vocab = {}
        self.dim = 200

    def encode(self,tokens:str):
        if isinstance(tokens,list):
            return [self.decode(token) for token in tokens]
        return self.model.encode(tokens).tolist()

    def __add__(self,others):
        if isinstance(others,tuple):
            key,value = others
            self.vocab[key] = self.encode(value)
        elif isinstance(others,str):
            self.vocab[others] = self.encode(others)
        return self

    def to_file(self,path):
        dir,_ = os.path.split(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(path,"w",encoding="utf-8") as f:
            f.write(f"{len(self.vocab)} {self.dim}\n")
            for key in self.vocab:
                f.write(f"{key} {' '.join(str(i) for i in self.vocab[key])}\n")
        return self