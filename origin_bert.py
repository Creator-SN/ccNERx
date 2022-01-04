from seqeval.metrics import classification_report, f1_score, precision_score, accuracy_score, recall_score
from transformers import Trainer, BertForTokenClassification, TrainingArguments
from CC.loaders.utils.vocab import Vocab
import json
from typing import Any
import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import EvalPrediction
from transformers import BertTokenizer
from torch import nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import argparse

parser = argparse.ArgumentParser(description="bert arguments")
parser.add_argument("dataset",type=str)
parser.add_argument("scale",type=int)
global_args = parser.parse_args()

class SimpleDataset(Dataset):
    def __init__(self, path, tokenizer: BertTokenizer, padding_length=256, tokens2ids: Any = None):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"load {path}"):
                data = json.loads(line.strip())
                text, labels = data["text"][:padding_length -
                                            2], data["label"][:padding_length-2]
                ids = tokenizer(''.join(
                    text), max_length=padding_length, padding="max_length", truncation=True, return_tensors="pt")
                for key in ids:
                    ids[key] = ids[key][0]
                labels_ids = tokens2ids(labels)
                labels = torch.tensor(
                    [-100]+labels_ids+[-100]*(padding_length-1-len(labels_ids)), dtype=torch.long)
                ids["labels"] = labels
                self.data.append(ids)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

report_path = f"data_record/{global_args.dataset}_bert_{global_args.scale}/"

args = {
    "train_file": f"data/few_shot/{global_args.dataset}/train_{global_args.scale}.json",
    "eval_file": f"data/few_shot/{global_args.dataset}/dev.json",
    "tag_file": f"data/few_shot/{global_args.dataset}/labels.txt",
}

tokenizer = BertTokenizer.from_pretrained("model/chinese_wwm_ext")


label_vocab = Vocab().from_files([args["tag_file"]])

train_set = SimpleDataset(
    args["train_file"], tokenizer, tokens2ids=label_vocab.token2id)
eval_set = SimpleDataset(
    args["eval_file"], tokenizer, tokens2ids=label_vocab.token2id)

train_arguments = TrainingArguments(
    output_dir=f"./runs/{global_args.dataset}/{global_args.scale}",
    do_eval=True,
    do_train=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    warmup_steps=190,
    logging_strategy="epoch",
    save_strategy="epoch",
    no_cuda=False,
)

def generate_csv(path,data):
    os.makedirs(path,exist_ok=True)
    with open(os.path.join(path,"eval.csv"),"w",encoding="utf-8") as f:
        f.write(f"f1,precision,recall,accuracy\n")
        for line in data:
            f.write(f'{line["f1"]},{line["precision"]},{line["recall"]},{line["accuracy"]}\n')
        f.flush()

def get_item(obj):
    if isinstance(obj,dict):
        for k in obj:
            obj[k] = get_item(obj[k])
    else:
        if getattr(obj,"item"):
            return obj.item()
    return obj

def generate_report(path,epoch,data):
    data = get_item(data)
    path = os.path.join(path,"reports")
    os.makedirs(path,exist_ok=True)
    with open(os.path.join(path,f"{epoch}_epoch.json"),"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

data_collections = []

def compute_metrics(predictions: EvalPrediction):
    global label_vocab,data_collections,report_path
    epoch = len(data_collections)+1
    labels = predictions.label_ids
    preds = predictions.predictions.argmax(-1)
    preds_list = []
    trues_list = []
    for label, pred in zip(labels, preds):
        label = torch.tensor(label)
        pred = torch.tensor(pred)
        mask = label.gt(-1)
        label = label[mask].tolist()
        pred = pred[mask].tolist()
        assert len(label) == len(pred)
        preds_list.append(list(i.replace("M-", "I-")
                          for i in label_vocab.id2token(pred)))
        trues_list.append(list(i.replace("M-", "I-")
                          for i in label_vocab.id2token(label)))
    result = {
        "accuracy": accuracy_score(trues_list, preds_list),
        "recall": recall_score(trues_list, preds_list),
        "f1": f1_score(trues_list, preds_list),
        "precision": precision_score(trues_list, preds_list),
        # "reports": classification_report(trues_list, preds_list, output_dict=True)
    }

    data_collections.append(result)
    generate_report(report_path,epoch,classification_report(trues_list, preds_list, output_dict=True))

    return result

def model_init():
    
    model = BertForTokenClassification.from_pretrained("model/chinese_wwm_ext", num_labels=len(label_vocab),
                                                   id2label=dict(
                                                       zip(range(len(label_vocab)), label_vocab.idx2item)),
                                                   label2id=label_vocab.item2idx)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    return model

trainer = Trainer(
    model_init=model_init,
    args=train_arguments,
    train_dataset=train_set,
    eval_dataset=eval_set,
    compute_metrics=compute_metrics
)

trainer.train()

generate_csv(report_path,data_collections)