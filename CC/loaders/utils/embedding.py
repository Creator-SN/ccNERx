from . import *
import numpy as np
from tqdm import *
import os
import pickle


class WordEmbedding():
    def __init__(self):
        self.embedding_index = {}
        self.dimension = -1
        self.reader = None

    def build_from_txt(self,embedding_path: str,max_scan_num: int = 1000000,add_seg_vocab: bool = False):
        self.reader = FileReader(embedding_path)
        line_totals = self.reader.line_size()
        if add_seg_vocab or max_scan_num < 0:
            max_scan_num = line_totals
        else:
            max_scan_num = min(max_scan_num, line_totals)
        for index in tqdm(range(max_scan_num),desc="load word embedding...",
                                total=max_scan_num):
            line = self.reader.line(index)
            line = line.strip().split()
            if index == 0:
                self.dimension = int(line[1])
            elif len(line) == 201:
                self.embedding_index[line[0]] = index
            elif len(line) > 201:
                print(f"{line} length more than 201")
                self.embedding_index[" ".join(line[:-200])] = index
            else:
                print(f"{line} embedding error")
        return self

    def get_embedding(self):
        return self.embedding_index, self.dimension, self.reader


class VocabEmbedding():
    def __init__(self, vocab: Vocab, cache_dir: str = "./temp/"):
        self.vocab: Vocab = vocab
        self.cache_dir: str = cache_dir
        self.dimension: int = 200

    def build_from_file(self,
                        embedding_path: str,
                        max_scan_num: int = 1000000,
                        add_seg_vocab: bool = False):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        cache_path = os.path.join(self.cache_dir,
                                  f"save_word_embedding_{max_scan_num}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.embedding, self.dimension = pickle.load(f)
            return self
        embedding_index = {}
        if embedding_path is not None:
            embedding_index, self.dimension, embedding_reader = WordEmbedding() \
                .build_from_txt(embedding_path,max_scan_num=max_scan_num,add_seg_vocab=add_seg_vocab).get_embedding()
        self.embedding = np.empty([self.vocab.size, self.dimension])
        if embedding_reader is not None:
            for idx, word in tqdm(enumerate(self.vocab.idx2item),
                                  desc="load vocab embedding"):
                if word in embedding_index:
                    line = embedding_reader.line(embedding_index[word])
                    line = line.strip().split()
                    np_embedding = np.empty(self.dimension)
                    np_embedding = line[-200:]
                    self.embedding[idx, :] = np_embedding
                else:
                    self.embedding[idx, :] = self.random_embedding()
        with open(cache_path, "wb") as f:
            pickle.dump((self.embedding, self.dimension), f, protocol=4)
        return self

    def random_embedding(self) -> np.numarray:
        scale = np.sqrt(3.0 / self.dimension)
        return np.random.uniform(-scale, scale, (1, self.dimension))

    def get_embedding(self):
        return self.embedding, self.dimension
