import os
from typing import Dict, List, OrderedDict, Set, Tuple
from tqdm import *
from collections import Counter, defaultdict
import json


class LabelCounter():

    def __init__(self, label_file, debug=False) -> None:
        self.label_type = "BIO"
        labels: Set[str] = set()
        with open(label_file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="loader labels..."):
                line = line.strip()
                if line[0] == "M" or line[0] == "S":
                    self.label_type = "BMES"
                labels.add("".join(line.split("-")[1:]))
        self.labels = labels
        self.counter: Counter = Counter()
        self.labels_instances: Dict[List[str]] = defaultdict(list)

    def __repr__(self) -> str:
        return f"""counter:{json.dumps(dict(sorted(zip(self.counter.keys(),self.counter.values()))), ensure_ascii=False, indent=4)}
        """
        """
        instances:{json.dumps(self.labels_instances,ensure_ascii=False,indent=4)}
        """

    def __add__(self, others: Tuple[str, str]) -> None:
        if not isinstance(others, tuple):
            raise TypeError(f"others must be a tuple")
        label, value = others
        if isinstance(label, list):
            label = str(label)
        if label not in self.labels:
            raise KeyError(f"label {label} not found")
        self.counter[label] += 1
        self.labels_instances[label].append(value)
        return self

    def sorted_keys(self):
        self.counter = Counter(dict(
            sorted(zip(self.counter.keys(), self.counter.values()))))
        return self

    def keys(self):
        return self.counter.keys()

    def items(self):
        return self.counter.items()

    def values(self):
        return self.counter.values()

    def add(self, labels: List[str], text: List[str]) -> None:
        if isinstance(text, str):
            text = list(text)
        start = -1
        end = -1
        labels.append("B-")
        last_label = None
        for index, label in enumerate(labels):
            if label.startswith("O") or label.startswith("B-") or label.startswith("S-"):
                if start != -1:
                    end = index
                    self += (last_label, text[start:end])
                start = -1
            if label.startswith("B-"):
                last_label = "".join(label.split("-")[1:])
                start = index
        return self
