from typing import List
import os

def merge_datasets(merge_list: List[str], output: str) -> None:
    dirname = os.path.dirname(output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(output, "w", encoding="utf-8") as saved_file:
        for dataset in merge_list:
            with open(dataset, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    saved_file.write(f"{line}\n")
            saved_file.flush()


def merge_labels(merge_list: List[str], output: str) -> None:
    dirname = os.path.dirname(output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    labels_set = set()
    labels = []
    for dataset in merge_list:
        with open(dataset, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line not in labels_set:
                    labels_set.add(line)
                    labels.append(line)
    with open(output, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")
        f.flush()
