
from CC.loaders.utils import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor
from transformers import BertTokenizer
from tqdm import *
from typing import *
from CC.loaders.utils.label import get_labels
from ICCSupervised.ICCSupervised import IDataLoader
import json
import numpy as np
import random


class LabelLXLoader(IDataLoader):
    def __init__(self, **args):
        KwargsParser(debug=True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("eval_batch_size", int, defaultValue=16) \
            .add_argument("test_batch_size", int, defaultValue=16) \
            .add_argument("word_embedding_file", str) \
            .add_argument("word_vocab_file", str) \
            .add_argument("word_vocab_file_with_tag", str) \
            .add_argument("train_file", str) \
            .add_argument("eval_file", str) \
            .add_argument("test_file", str) \
            .add_argument("tag_file", str) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("output_eval", bool, defaultValue=False) \
            .add_argument("max_scan_num", int, defaultValue=1000000) \
            .add_argument("add_seq_vocab", bool, defaultValue=False) \
            .add_argument("max_seq_length", int, defaultValue=256) \
            .add_argument("max_word_num", int, defaultValue=5) \
            .add_argument("default_tag", str, defaultValue="O") \
            .add_argument("use_test", bool, defaultValue=False) \
            .add_argument("do_shuffle", bool, defaultValue=False) \
            .add_argument("do_predict", bool, defaultValue=False) \
            .add_argument("lexicon_tree_cache_path", str, optional=True) \
            .add_argument("word_vacab_cache_path", str, optional=True) \
            .add_argument("task_name", str) \
            .add_argument("tag_rules", dict) \
            .add_argument("debug", bool, defaultValue=False) \
            .parse(self, **args)

        self.read_data_set()
        self.verify_data()
        self.process_data()

    def __restore_attr(self, attr_name: str, construct, restore_path: str = None):
        if restore_path is None:
            # If the restore path is not set, directly construct
            setattr(self, attr_name, construct())
        else:
            # If restore file exists, restore it
            if os.path.exists(restore_path):
                with open(restore_path, "rb") as f:
                    setattr(self, attr_name, pickle.load(f))
            else:
                obj = construct()
                setattr(self, attr_name, obj)
                # save cache file for restore
                with open(restore_path, "wb") as f:
                    pickle.dump(obj, f)
        return self

    def read_data_set(self):
        self.data_files = [self.train_file, self.eval_file, self.test_file]
        # loading lexicon tree

        self.__restore_attr("lexicon_tree", lambda: TrieFactory.get_trie_from_vocabs(
            [self.word_vocab_file], self.max_scan_num), getattr(self, "lexicon_tree_cache_path", None))

        self.matched_words: List[str] = TrieFactory.get_all_matched_word_from_dataset(
            self.data_files, self.lexicon_tree)
        # restore all word_vocab_file_with_tag
        self.__restore_attr("word_vocab", lambda: VocabTag().from_files(
            [self.word_vocab_file_with_tag], is_word=True, has_default=False, unk_num=5, max_scan_num=self.max_scan_num), getattr(self, "word_vocab_cache_path", None))

        matched_words_with_tags = [(word, self.word_vocab.tag(word))
                                   for word in self.matched_words]
        # re assign word_vocab for matched_words
        self.word_vocab = VocabTag().from_list(matched_words_with_tags,
                                               is_word=True, has_default=False, unk_num=5)

        self.tag_vocab = Vocab().from_files([self.tag_file])

        self.vocab_embedding, self.embedding_dim = VocabEmbedding(self.word_vocab, cache_dir=f"./temp/{self.task_name}").build_from_file(
            self.word_embedding_file, self.max_scan_num, self.add_seq_vocab).get_embedding()

        self.tag_convert: TagConvert = TagConvert(self.tag_rules)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

    def verify_data(self):
        pass

    def process_data(self):
        if self.use_test:
            self.myData_test = LEXBertDataSet(self.data_files[2], **vars(self))
            self.dataiter_test = DataLoader(
                self.myData_test, batch_size=self.test_batch_size)
        else:
            self.myData = LEXBertDataSet(self.data_files[0], **vars(self))
            self.dataiter = DataLoader(self.myData, batch_size=self.batch_size)

    def __call__(self):
        if self.use_test:
            return {
                'test_set': self.myData_test,
                'test_iter': self.dataiter_test,
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


class LEXBertDataSet(Dataset):
    def __init__(self, dataset_file: str, **args):
        self.file = dataset_file
        for name in args.keys():
            setattr(self, name, args[name])
        self.__init_dataset()

    def to_numpy(self, input_ids, token_type_ids, attention_mask, label_ids, input_labels, input_origin_labels):
        np_input_ids = np.zeros(self.max_seq_length, dtype=np.int)
        np_input_ids[:len(input_ids)] = input_ids

        np_token_type_ids = np.ones(self.max_seq_length, dtype=np.int)
        np_token_type_ids[:len(token_type_ids)] = token_type_ids

        np_attention_mask = np.ones(self.max_seq_length, dtype=np.int)
        np_attention_mask[:len(attention_mask)] = attention_mask

        np_labels_ids = np.zeros(self.max_seq_length, dtype=np.int)
        np_labels_ids[:len(label_ids)] = label_ids

        np_input_labels = np.zeros(self.max_seq_length, dtype=np.int)
        np_input_labels[:len(input_labels)] = input_labels
        np_input_labels[len(input_labels):] = -100

        np_input_origin_labels = np.zeros(self.max_seq_length, dtype=np.int)
        np_input_origin_labels[:len(input_origin_labels)] = input_origin_labels
        np_input_origin_labels[len(input_origin_labels):] = -100

        return np_input_ids, np_token_type_ids, np_attention_mask,  np_input_labels, np_input_origin_labels, np_labels_ids

    def combine_prompt(self, data_text, data_labels, prompts):
        # clone origin data
        data_text = ["[CLS]"] + data_text[:self.max_seq_length-2] + ["[SEP]"]
        data_labels = [self.default_tag] + \
            data_labels[:self.max_seq_length-2] + [self.default_tag]
        origin_text = data_text[:]

        labels = [-100] * len(data_text)
        attention_mask = [1] * len(data_text)
        token_type_ids = [0] * len(data_text)
        for prompt, mask, label, origin in prompts:
            if len(prompt)+len(data_text) <= self.max_seq_length:
                data_text += prompt
                attention_mask += mask
                data_labels += label
                origin_text += origin
                labels += [self.tokenizer.convert_tokens_to_ids(
                    [ch])[0] if mask_bit == 0 else -100 for ch, mask_bit in zip(origin, mask)]
        data_text[-1] = "[SEP]"
        origin_text[-1] = "[SEP]"
        input_ids = self.tokenizer.convert_tokens_to_ids(data_text)
        labels_ids = self.tag_vocab.token2id(data_labels)
        origin_labels = self.tokenizer.convert_tokens_to_ids(origin_text)
        return self.to_numpy(input_ids, token_type_ids, attention_mask, labels_ids, labels, origin_labels)

    def convert_embedding(self, item):
        if "text" not in item:
            raise KeyError(f"key text not exists in item: {item}")
        if "label" not in item:
            raise KeyError(f"key label not exists in item: {item}")
        if "replace" not in item:
            raise KeyError(f"key replace not exists in item: {item}")

        replace_set = {}
        for span in item["replace"]:
            replace_set[f"{span['start']},{span['end']}"] = span["origin"]

        entity_collections = get_entities(item["label"], item["text"])

        # resolve prompt
        exist_prompt = set()
        for start, end, label, word in entity_collections:
            # prompt,mask,labels,origin_prompt
            prompt = self.tag_convert.tag2prompt(
                get_labels(label, len(word)), word)
            key = hash(str(prompt[3]))
            prompts = [prompt]
            if key not in exist_prompt:
                exist_prompt.add(key)
                # check the origin word
                if f"{start},{end}" in replace_set:
                    origin = replace_set[f"{start},{end}"]
                    temp_prompt = word + list("应该是") + ['[MASK]'] * \
                        len(origin) + [","]
                    temp_mask = [1] * (3+len(word)) + [0] * len(origin) + [1]
                    temp_labels = get_labels(label, len(
                        word)) + [self.default_tag] * 3 + get_labels(label, len(origin)) + [self.default_tag]
                    temp_origin = word + list("应该是") + origin + [","]
                    prompts.append((temp_prompt, temp_mask,
                                   temp_labels, temp_origin))

                yield self.combine_prompt(item["text"], item["label"], prompts)
        matched_words = self.lexicon_tree.getAllMatchedWordList(
            item["text"], self.max_word_num)

        # for matched_word add prompt
        for words in matched_words:
            for word in words:
                tag = self.word_vocab.tag(word)
                # if the word tag is "O", skip...
                if tag[0] == self.default_tag:
                    continue
                prompt = self.tag_convert.tag2prompt(
                    tag, word)
                key = hash(str(prompt[3]))
                if key not in exist_prompt:
                    exist_prompt.add(key)
                    prompts = [prompt]
                    yield self.combine_prompt(item["text"], item["label"], prompts)

    def __init_dataset(self):
        line_total = FileUtil.count_lines(self.file)
        self.input_token_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.origin_labels = []
        self.input_labels = []
        self.labels = []
        for line in tqdm(FileUtil.line_iter(self.file), desc=f"load dataset from {self.file}", total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            for input_token_ids, token_type_ids, attention_mask, input_labels, origin_label, labels in self.convert_embedding(
                    data):
                self.input_token_ids.append(input_token_ids)
                self.token_type_ids.append(token_type_ids)
                self.attention_mask.append(attention_mask)
                self.origin_labels.append(origin_label)
                self.input_labels.append(input_labels)
                self.labels.append(labels)
        self.size = len(self.input_token_ids)
        self.input_token_ids = np.array(self.input_token_ids)
        self.token_type_ids = np.array(self.token_type_ids)
        self.attention_mask = np.array(self.attention_mask)
        self.origin_labels = np.array(self.origin_labels)
        self.input_labels = np.array(self.input_labels)
        self.labels = np.array(self.labels)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        return {
            'input_ids': tensor(self.input_token_ids[idx]),
            'attention_mask': tensor(self.attention_mask[idx]),
            'token_type_ids': tensor(self.token_type_ids[idx]),
            'origin_labels': tensor(self.origin_labels[idx]),
            'input_labels': tensor(self.input_labels[idx]),
            'labels': tensor(self.labels[idx])
        }

    def __len__(self):
        return self.size
