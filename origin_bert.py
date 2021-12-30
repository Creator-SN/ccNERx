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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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


args = {
    "train_file": "data/few_shot/weibo/train_1350.json",
    "eval_file": "data/few_shot/weibo/dev.json",
    "tag_file": "data/few_shot/weibo/labels.txt",
}

tokenizer = BertTokenizer.from_pretrained("model/chinese_wwm_ext")


label_vocab = Vocab().from_files([args["tag_file"]])

train_set = SimpleDataset(
    args["train_file"], tokenizer, tokens2ids=label_vocab.token2id)
eval_set = SimpleDataset(
    args["eval_file"], tokenizer, tokens2ids=label_vocab.token2id)




train_arguments = TrainingArguments(
    output_dir="./save_models/weibo/1350_256padding",
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


def compute_metrics(predictions: EvalPrediction):
    global label_vocab
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
    # print(classification_report(trues_list,preds_list))
    return {
        "accuracy": accuracy_score(trues_list, preds_list),
        "recall": recall_score(trues_list, preds_list),
        "f1": f1_score(trues_list, preds_list),
        "precisition": precision_score(trues_list, preds_list),
        "reports": classification_report(trues_list, preds_list, output_dict=True)
    }

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
