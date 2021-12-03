import torch
from torch._C import dtype
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


class PTLoaderV2(IDataLoader):
    def __init__(self, **args):
        KwargsParser(debug=True) \
            .add_argument("batch_size", int, defaultValue=4) \
            .add_argument("train_file", str) \
            .add_argument("tag_file", str) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("max_seq_length", int, defaultValue=256) \
            .add_argument("do_shuffle", bool, defaultValue=False) \
            .add_argument("task_name", str) \
            .add_argument("tag_rules", dict) \
            .add_argument("debug", bool, defaultValue=False) \
            .add_argument("pass_none_rule", bool, defaultValue=False) \
            .parse(self, **args)

        self.read_data_set()
        self.process_data()

    def read_data_set(self):

        # self.max_tag_length = max(len(i) for i in self.tag_rules.values())

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_file)

        self.tag_rules_to_unused = dict(
            zip(self.tag_rules.keys(), list(range(1, len(self.tag_rules)+1))))

    def process_data(self):
        self.myData = PTLoaderV2DataSet(self.train_file,
                                        self.tokenizer,
                                        self.tag_rules,
                                        self.max_seq_length,
                                        self.tag_rules_to_unused)
        self.dataiter = DataLoader(self.myData, batch_size=self.batch_size,shuffle=self.do_shuffle)

    def __call__(self):
        return {'train_set': self.myData, 'train_iter': self.dataiter}


class PTLoaderV2DataSet(Dataset):
    def __init__(self,
                 file: str,
                 tokenizer: BertTokenizer,
                 tag_rules: Dict[str, str],
                 max_seq_length: int,
                 #  max_tag_length: int,
                 tag_rules_to_unused,
                 default_tag="O",
                 do_shuffle: bool = False):
        self.__dict__.update(locals().items())
        self.__init_dataset()

    def convert_prompts(self, item):
        if "text" not in item:
            raise KeyError(f"key text not exists in item: {item}")
        if "label" not in item:
            raise KeyError(f"key label not exists in item: {item}")
        text, label = item["text"][:self.max_seq_length - 2], item["label"]

        ids = self.tokenizer(''.join(text))

        tag_length = 1

        cur_length = len(ids["input_ids"])
        offset: int = self.max_seq_length
        padding = tag_length + 2

        max_padding = 512

        text_ids = {
            "prompt_input_ids": torch.zeros(max_padding, dtype=torch.int),
            "prompt_attention_mask": torch.zeros(max_padding, dtype=torch.int),
            "prompt_token_type_ids": torch.zeros(max_padding, dtype=torch.int),
            "prompt_labels": torch.tensor([-100] * max_padding,
                                          dtype=torch.int),
            "prompt_origin_labels": None,
        }
        text_ids["prompt_input_ids"][:cur_length] = torch.tensor(
            ids["input_ids"], dtype=torch.int)

        text_ids["prompt_attention_mask"][:cur_length] = 1

        text_ids["prompt_token_type_ids"][cur_length:] = 1

        text_ids["prompt_origin_labels"] = text_ids["prompt_input_ids"].clone()

        current = dict(
            zip(text_ids.keys(), [i.clone() for i in text_ids.values()]))

        collections = {}
        for key in current.keys():
            collections[key] = []

        # [512 / 7] * 4
        index_length = math.ceil(
            (max_padding - self.max_seq_length) /
            (3 + tag_length)) * tag_length
        # index_length = 300

        indexes = torch.zeros(4 * self.max_seq_length, dtype=np.int)
        index_offset = 0
        cur_length = offset
        count = 0
        for i in range(self.max_seq_length):
            prompt_text = list(f"{i}是") + ["[MASK]"] * tag_length + [
                ","
            ]
            if cur_length + len(prompt_text) >= max_padding:
                count += 1
                # 超出长度判断，生成当前组数据
                for key in collections.keys():
                    collections[key].append(current[key])
                # 重置数据，为下一组数据做准备
                cur_length = offset
                current = dict(
                    zip(text_ids.keys(),
                        [i.clone() for i in text_ids.values()]))
            # 左闭右开
            start, end = cur_length + len(str(i)) + 1, cur_length + len(
                str(i)) + padding - 1

            for j in range(start, end):
                indexes[index_offset] = count * max_padding + j
                index_offset += 1

            char_label = self.tag_rules_to_unused[
                label[i - 1].split("-")
                [-1]] if i != 0 and i - 1 < len(text) else self.tag_rules_to_unused[self.default_tag]

            current["prompt_input_ids"][cur_length:cur_length +
                                        len(prompt_text)] = torch.tensor(
                                            self.tokenizer.
                                            convert_tokens_to_ids(prompt_text),
                                            dtype=torch.int)

            current["prompt_attention_mask"][cur_length:cur_length +
                                             len(prompt_text)] = torch.tensor(
                                                 [1] * (len(str(i)) + 1) +
                                                 [0] * tag_length +
                                                 [1],
                                                 dtype=torch.int)
            mask_label_ids = [char_label]
            try:
                current["prompt_labels"][cur_length:cur_length +
                                         len(prompt_text)] = torch.tensor(
                    [-100] * (len(str(i)) + 1) +
                    mask_label_ids + [0] *
                    (tag_length -
                     len(mask_label_ids)) + [-100],
                    dtype=torch.int)
            except TypeError as e:
                print(mask_label_ids)
                raise e
            current["prompt_origin_labels"][
                cur_length:cur_length + len(prompt_text)] = torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(list(str(i) + '是')) +
                    mask_label_ids + [0] *
                    (tag_length - len(mask_label_ids)) +
                    self.tokenizer.convert_tokens_to_ids([","]))

            cur_length += len(prompt_text)

        # 最后一组数据
        for key in collections.keys():
            collections[key].append(current[key].clone())

        # 按序合并到一维
        for key in collections.keys():
            collections[key] = torch.stack(collections[key]).reshape(-1)

        collections["prompt_indexes"] = indexes

        return collections

    def __init_dataset(self):
        line_total = FileUtil.count_lines(self.file)

        self.dataset = []
        self.keys = [
            "input_ids", "attention_mask", "token_type_ids", "labels",
            "origin_labels"
        ]

        self.keys = [f"prompt_{i}" for i in self.keys]

        for line in tqdm(FileUtil.line_iter(self.file),
                         desc=f"load dataset from {self.file}",
                         total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            self.dataset.append(self.convert_prompts(data))

        self.size = len(self.dataset)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        if isinstance(idx, list):
            data = {}
            for key in self.keys:
                data[key] = [self.dataset[j][key] for j in idx]
                data[key] = torch.stack(data[key])
            return data
        return self.dataset[idx]
        # if isinstance(idx, list):
        #     return dict(
        #         zip(self.keys, [
        #             torch.stack([self.dataset[i][j] for i in idx])
        #             for j in range(len(self.keys))
        #         ]))
        # return dict(zip(self.keys, self.dataset[idx]))

    def __len__(self):
        return self.size
