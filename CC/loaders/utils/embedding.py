from CC.loaders.utils import *
import numpy as np
from tqdm import *
import os
import pickle


class WordEmbedding():
    def __init__(self):
        self.embedding = {}
        self.dimension = -1

    def build_from_txt(self, embedding_path: str, max_scan_num: int = 1000000, add_seg_vocab: bool = False):
        line_totals = FileUtil.count_lines(embedding_path,show_progress=True)
        if add_seg_vocab or max_scan_num < 0:
            max_scan_num = line_totals
        else:
            max_scan_num = min(max_scan_num, line_totals)
        for index, line in tqdm(enumerate(FileUtil.line_iter(embedding_path)), desc="load word embedding...", total=max_scan_num):
            if index > max_scan_num:
                break
            line = line.strip().split()
            if index == 0:
                assert len(line) == 2
                self.dimension = int(line[1])
            elif len(line) == 201:
                embedding = np.empty((1, 200))
                embedding = line[1:]
                self.embedding[line[0]] = embedding
            elif len(line) > 201:
                print(f"{line} length more than 201")
                embedding = np.empty((1, 200))
                embedding = line[-200:]
                self.embedding["".join(line[:-200])] = embedding
            else:
                print(f"{line} embedding error")
        return self

    def get_embedding(self):
        return self.embedding, self.dimension


class VocabEmbedding():
    def __init__(self, vocab: Vocab, cache_dir: str = "./temp/"):
        self.vocab: Vocab = vocab
        self.cache_dir: str = cache_dir
        self.dimension: int = 200

    def build_from_file(self, embedding_path: str, max_scan_num: int = 1000000, add_seg_vocab: bool = False):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        cache_path = os.path.join(
            self.cache_dir, f"save_word_embedding_{max_scan_num}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.embedding, self.dimension = pickle.load(f)
            return self
        embedding = {}
        if embedding_path is not None:
            embedding, self.dimension = WordEmbedding().build_from_txt(
                embedding_path).get_embedding()
        self.embedding = np.empty([self.vocab.size, self.dimension])
        for idx, word in tqdm(enumerate(self.vocab.idx2item), desc="load vocab embedding"):
            if word in embedding:
                self.embedding[idx, :] = embedding[word]
            else:
                self.embedding[idx, :] = self.random_embedding()
        with open(cache_path, "wb") as f:
            pickle.dump((self.embedding, self.dimension), f, protocol=4)
        return self

    def random_embedding(self) -> np.numarray:
        scale = np.sqrt(3.0/self.dimension)
        return np.random.uniform(-scale, scale, (1, self.dimension))

    def get_embedding(self):
        return self.embedding, self.dimension
