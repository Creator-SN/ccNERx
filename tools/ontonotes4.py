# https://aclanthology.org/N13-1006.pdf

import os
import xml.etree.ElementTree as ET
import re

scan_dir = ""

class Rule():
    def __init__(self,test=None,action=None) -> None:
        self.test = test
        self.action = action

labels = set()
expect_labels = set(["GPE","LOC","ORG","PERSON"])
occur_labels = set()

def scan_file(file_name,data,rules):
    global labels
    global expect_labels
    global occur_labels
    _,file = os.path.split(file_name)
    # scan file
    file_data = {"text":[],"label":[]}
    with open(file_name,"r",encoding="utf-8") as f:
        content = f.read()
        root = ET.fromstring(content)
        for child in root:
            inner = list(re.sub("[^\S\n]",'',child.text))
            if child.tag=="ENAMEX":
                real = child.attrib["TYPE"]
                labels.add(real)
                text = inner
                if real in expect_labels:
                    label = [f"I-{real}"] * len(text)
                    label[-1]=f"E-{real}"
                    if len(label)==1:
                        label[0] = f"S-{real}"
                    else:
                        label[0] = f"B-{real}"
                else:
                    label = ["O"] * len(text)
            else:
                text = inner
                label = ["O"] * len(text)
            occur_labels.update(label)
            file_data["text"]+=text
            file_data["label"]+=label
            tail = list(re.sub("[^\S\n]",'',child.tail))
            text = tail
            label = ["O"] * len(text)
            file_data["text"]+=text
            file_data["label"]+=label
    for rule in rules:
        if rule.test is not None and rule.test(file):
            rule.action(data,file_data)
            break

def scan(dir,data,rules):
    dir_list = os.listdir(dir)
    if "map.txt" in dir_list:
        with open(os.path.join(dir,"map.txt"),"r",encoding="utf-8") as f:
            for line in f:
                relative_path = line.strip().split()[0]+".name"
                path = os.path.join(dir,relative_path)
                if os.path.exists(path):
                    scan_file(path,data,rules)
    for file in dir_list:
        path = os.path.join(dir,file)
        if os.path.isdir(path):
            scan(path,data,rules)


eval_sent = 0

def eval_test_file(file):
    fileset = set([f"chtb_{i:04}.name" for i in range(1,326)])
    # Lattice extra
    fileset.update([f"chtb_{i:04}.name" for i in range(1001,1079)])
    return file in fileset

def eval_test_action(data,file_data):
    global eval_sent
    last = 0
    for index in range(len(file_data["text"])):
        if file_data["text"][index]=='\n':
            if not ''.join(file_data["text"][last:index]).startswith("（完）"):
                eval_sent+=1
                data["eval" if eval_sent%2==1 else "test"].append({
                    "text":file_data["text"][last:index],
                    "label":file_data["label"][last:index]
                })
            last = index+1
    if len(file_data["text"][last:])>0:
        if not ''.join(file_data["text"][last:]).startswith("（完）"):
            eval_sent+=1
            data["eval" if eval_sent%2==1 else "test"].append({
                "text":file_data["text"][last:],
                "label":file_data["label"][last:]
            })

def train_file(file):
    return True

def train_action(data,file_data):
    last = 0
    for index in range(len(file_data["text"])):
        if file_data["text"][index]=='\n':
            if not ''.join(file_data["text"][last:index]).startswith("（完）"):
                data["train"].append({
                    "text":file_data["text"][last:index],
                    "label":file_data["label"][last:index]
                })
            last = index+1
    if len(file_data["text"][last:])>0:
        if not ''.join(file_data["text"][last:]).startswith("（完）"):
            data["train"].append({
                "text":file_data["text"][last:],
                "label":file_data["label"][last:]
            })
            
import json
def to_json_file(file,data):
    dir,_ = os.path.split(file)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(file,"w",encoding="utf-8") as f:
        for line in data:
            f.write(f"{json.dumps(line,ensure_ascii=False)}\n")
        f.flush()

def to_file(file,data):
    dir,_ = os.path.split(file)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(file,"w",encoding="utf-8") as f:
        for line in data:
            f.write(f"{line}\n")
        f.flush()

def get_labels(data,occur=None):
    labels = ["O"]
    types = ["B","I","E","S"]
    for item in data:
        for t in types:
            if occur is None or f"{t}-{item}" in occur:
                labels.append(f"{t}-{item}")
    return labels

if __name__=="__main__":
    data = {
        "eval":[],
        "test":[],
        "train":[]
    }

    rules = [Rule(eval_test_file,eval_test_action),Rule(train_file,train_action)]
    scan("/data/ontonotes-release-4.0/data/files/data/chinese",data,rules)
    # print(data)
    import sys
    with open("./local_temp_code/temp.out","w",encoding="utf-8") as sys.stdout:
        for key in data:
            to_json_file(f"data/ontonotes4/{key}.json",data[key])
        print(occur_labels)
        to_file(f"data/ontonotes4/labels.txt",get_labels(expect_labels))