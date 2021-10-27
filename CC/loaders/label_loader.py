from logging import debug
from pickle import FALSE
from torch._C import Value
from transformers.file_utils import filename_to_url
from CC.loaders.utils.label_collections import LabelCollection
from ICCSupervised.ICCSupervised import IDataLoader
import os
from CC.loaders.utils import *
from tqdm import *
import json
import random
import math


class LabelLoader(IDataLoader):
    def __init__(self, **args):
        KwargsParser().add_argument("debug", bool, defaultValue=False) \
            .add_argument("file_name", str) \
            .add_argument("random_rate", float, defaultValue=1.0) \
            .add_argument("expansion_rate", int, defaultValue=1) \
            .add_argument("auto_loader",bool,defaultValue=True) \
            .parse(self, **args)
        if self.auto_loader:
            self.read_data_set(self.file_name, self.random_rate) \
                .process_data(self.expansion_rate)

    def __call__(self):
        return self.items

    def to_file(self, file_name):
        d = os.path.dirname(file_name)
        if not os.path.exists(d):
            os.mkdir(d)
        with open(file_name, "w", encoding="utf-8") as f:
            for text, label in self.items:
                line = json.dumps({
                    "text": text,
                    "label": label
                }, ensure_ascii=False)
                f.write(f"{line}\n")
        return self

    def read_data_set(self, file_name: str, random_rate: float = 1.0):
        count_lines = FileUtil.count_lines(file_name)
        pick_count = math.floor(count_lines * random_rate)
        self.items = []
        for line in tqdm(FileUtil.line_iter(file_name), total=count_lines):
            line = line.strip()
            data = json.loads(line)
            if "text" not in data:
                raise KeyError("text not exists")
            if "label" not in data:
                raise KeyError("label not exists")
            self.items.append((data["text"], data["label"]))
        self.items: List[Tuple[str, str]] = random.sample(
            self.items, pick_count)
        self.labels_collections: LabelCollection = LabelCollection()
        for text, label in self.items:
            self.labels_collections += (text, label)
        return self

    def verify_data(self):
        return self

    def process_data(self, expansion_rate: int = 1) :
        if getattr(self, "items", None) is None:
            raise ValueError("run read_data_set firstly!")
        new_items = []
        sample_set = set()
        debug_sample = 10

        def key(item):
            return hash(f"{str(item[0])}_{str(item[1])}")
        for text, label in self.items:
            # add sample first
            count = 0
            k = key((text, label))
            if k not in sample_set:
                count += 1
                sample_set.add(k)
                new_items.append((text, label))
            # important: reverse
            spans = self.labels_collections.get_label_slice(
                text, label)
            spans.reverse()
            repeat_count = expansion_rate*10
            while count < expansion_rate and repeat_count > 0:
                repeat_count -= 1
                new_text = text[:]
                new_label = label[:]
                for span in spans:
                    same_label_texts = self.labels_collections[span.label]
                    sample_text = random.choice(same_label_texts)
                    # replace
                    new_text[span.start:span.end] = sample_text
                    temp_label = [
                        f"I-{span.label}" for _ in sample_text]
                    temp_label[0] = f"B-{span.label}"
                    temp_label[-1] = new_label[span.end-1]
                    new_label[span.start:span.end] = temp_label
                    assert len(new_text) == len(
                        new_label), f"text:{new_text}\nlabel:{new_label}"
                k = key((new_text, new_label))
                if k not in sample_set:
                    if self.debug and debug_sample > 0:
                        print(new_text, new_label)
                        debug_sample -= 1
                    count += 1
                    sample_set.add(k)
                    new_items.append((new_text, new_label))
        self.items = new_items
        return self
