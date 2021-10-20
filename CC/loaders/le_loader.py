from CC.loaders.utils import *
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import tensor
from transformers import BertTokenizer
from tqdm import *
from typing import *
from ICCSupervised.ICCSupervised import IDataLoader
import json
import numpy as np
import random
from distutils.util import strtobool


class LLoader(IDataLoader):
    def __init__(self, **args):
        KwargsParser(True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("eval_batch_size", int, defaultValue=16) \
            .add_argument("word_embedding_file", str) \
            .add_argument("word_vocab_file", str) \
            .add_argument("train_file", str) \
            .add_argument("eval_file", str) \
            .add_argument("test_file", str) \
            .add_argument("tag_file", str) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("output_eval", bool, defaultValue=True) \
            .add_argument("max_scan_num", int, defaultValue=1000000) \
            .add_argument("add_seq_vocab", bool, defaultValue=False) \
            .add_argument("max_seq_length", int, defaultValue=256) \
            .add_argument("max_word_num", int, defaultValue=5) \
            .add_argument("deafult_tag", str, defaultValue="O") \
            .add_argument("use_test", bool, defaultValue=False) \
            .parse(self, **args)

        self.read_data_set()
        self.verify_data()
        self.process_data(self.batch_size, self.eval_batch_size)

    def read_data_set(self):
        # build lexicon tree
        self.lexicon_tree: Trie = TrieFactory.get_trie_from_vocabs(
            [self.word_vocab_file], self.max_scan_num)

        self.data_files: List[str] = [
            self.train_file, self.eval_file, self.test_file]

        self.matched_words: List[str] = TrieFactory.get_all_matched_word_from_dataset(
            self.data_files, self.lexicon_tree)

        self.word_vocab: Vocab = Vocab().from_list(
            self.matched_words, is_word=True, has_default=False, unk_num=5)

        self.tag_vocab: Vocab = Vocab().from_files(
            [self.tag_file], is_word=False)

        self.vocab_embedding, self.embedding_dim = VocabEmbedding(self.word_vocab).build_from_file(
            self.word_embedding_file, self.max_scan_num, self.add_seq_vocab).get_embedding()

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

    def verify_data(self):
        pass

    def process_data(self, batch_size: int, eval_batch_size: int = None):
        if self.use_test:
            self.myData_test = LEBertDataSet(self.data_files[2], self.tokenizer, self.lexicon_tree,
                                             self.word_vocab, self.tag_vocab, self.max_word_num, self.max_seq_length, self.default_tag)
            self.dataiter = DataLoader(self.myData_test, batch_size=batch_size)
        else:
            self.myData = LEBertDataSet(self.data_files[0], self.tokenizer, self.lexicon_tree, self.word_vocab,
                                        self.tag_vocab, self.max_word_num, self.max_seq_length, self.default_tag)
            self.dataiter = DataLoader(self.myData, batch_size=batch_size)
            if self.output_eval:
                self.myData_eval = LEBertDataSet(self.data_files[1], self.tokenizer, self.lexicon_tree, self.word_vocab,
                                                 self.tag_vocab, self.max_word_num,  self.max_seq_length, self.default_tag)
                self.dataiter_eval = DataLoader(
                    self.myData_eval, batch_size=eval_batch_size)

    def __call__(self):
        if self.use_test:
            return {
                'test_set': self.myData_test,
                'test_iter': self.myData_test,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }
        if self.output_eval:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'eval_set': self.myData_eval,
                'eval_iter': self.dataiter_eval,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }
        else:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }


class LEBertDataSet(Dataset):
    def __init__(self, file: str, tokenizer, lexicon_tree: Trie, word_vocab: Vocab, tag_vocab: Vocab, max_word_num: int, max_seq_length: int, default_tag: str, do_shuffle: bool = False):
        self.file: str = file
        self.tokenizer = tokenizer
        self.lexicon_tree: Trie = lexicon_tree
        self.word_vocab: Vocab = word_vocab
        self.label_vocab: Vocab = tag_vocab
        self.max_word_num: int = max_word_num
        self.max_seq_length: int = max_seq_length
        self.default_tag: str = default_tag
        self.do_shuffle: bool = do_shuffle
        self.__init_dataset()

    def __init_dataset(self):
        line_total = FileUtil.count_lines(self.file)
        self.input_token_ids = []
        self.segment_ids = []
        self.attention_mask = []
        self.matched_word_ids = []
        self.matched_word_mask = []
        self.labels = []

        for line in tqdm(FileUtil.line_iter(self.file), desc=f"load dataset from {self.file}", total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            assert "text" in data and "label" in data, "dataset json type error"
            text = ["[CLS]"] + data["text"][:self.max_seq_length-2] + ["[SEP]"]
            label = [self.default_tag] + \
                data["label"][:self.max_seq_length-2] + [self.default_tag]
            token_ids = self.tokenizer.convert_tokens_to_ids(text)
            label_ids = self.label_vocab.token2id(label)

            labels = np.zeros(self.max_seq_length, dtype=np.int)
            labels[:len(label_ids)] = label_ids
            # init input
            input_token_ids = np.zeros(self.max_seq_length, dtype=np.int)
            input_token_ids[:len(token_ids)] = token_ids
            segment_ids = np.ones(self.max_seq_length, dtype=np.int)
            segment_ids[:len(token_ids)] = 0
            attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
            attention_mask[:len(token_ids)] = 1
            matched_word_ids = np.zeros(
                (self.max_seq_length, self.max_word_num), dtype=np.int)
            matched_word_mask = np.zeros(
                (self.max_seq_length, self.max_word_num), dtype=np.int)
            # get matched word
            matched_words = self.lexicon_tree.getAllMatchedWordList(
                text, self.max_word_num)
            for i, words in enumerate(matched_words):
                word_ids = self.word_vocab.token2id(words)
                matched_word_ids[i][:len(word_ids)] = word_ids
                matched_word_mask[i][:len(word_ids)] = 1

            assert input_token_ids.shape[0] == segment_ids.shape[0]
            assert input_token_ids.shape[0] == attention_mask.shape[0]
            assert input_token_ids.shape[0] == matched_word_ids.shape[0]
            assert input_token_ids.shape[0] == matched_word_mask.shape[0]
            assert input_token_ids.shape[0] == labels.shape[0]
            assert matched_word_ids.shape[1] == matched_word_mask.shape[1]

            self.input_token_ids.append(input_token_ids)
            self.segment_ids.append(segment_ids)
            self.attention_mask.append(attention_mask)
            self.matched_word_ids.append(matched_word_ids)
            self.matched_word_mask.append(matched_word_mask)
            self.labels.append(labels)
        self.size = len(self.input_token_ids)
        self.input_token_ids = np.array(self.input_token_ids)
        self.segment_ids = np.array(self.segment_ids)
        self.attention_mask = np.array(self.attention_mask)
        self.matched_word_ids = np.array(self.matched_word_ids)
        self.matched_word_mask = np.array(self.matched_word_mask)
        self.labels = np.array(self.labels)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        return {
            'input_ids': tensor(self.input_token_ids[idx]),
            'attention_mask': tensor(self.attention_mask[idx]),
            'token_type_ids': tensor(self.segment_ids[idx]),
            'matched_word_ids': tensor(self.matched_word_ids[idx]),
            'matched_word_mask': tensor(self.matched_word_mask[idx]),
            'labels': tensor(self.labels[idx])
        }

    def __len__(self):
        return self.size
