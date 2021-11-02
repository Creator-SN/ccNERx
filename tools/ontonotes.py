#%%
import os
import xml.etree.ElementTree as ET
import re
data_dir = "data/ontonotes-release-5.0/data/files/data/chinese/annotations"
default_label = "O"
save_file = "temp/ontonotes-name.bmes"
with open(save_file,"w",encoding="utf-8") as save:
    for d in os.listdir(data_dir):
        d = os.path.join(data_dir,d)
        dir_list = os.listdir(d)
        if "map.txt" in dir_list:
            m = os.path.join(d,"map.txt")
            with open(m,"r",encoding="utf-8") as f:
                line = f.readline()
                while line:
                    filename = line.split()[0]
                    fullpath = os.path.join(d,f"{filename}.name")
                    if os.path.exists(fullpath):
                        print(f"open {fullpath}")
                        text = []
                        labels = []
                        with open(fullpath,"r",encoding="utf-8") as file:
                            content = file.read()
                            root = ET.fromstring(content)
                            t = list(re.sub("[^\S\n]",'',root.text))
                            l = [default_label for i in t]
                            text += t
                            labels += l
                            for child in root:
                                t = list(re.sub("[^\S\n]","",child.text))
                                if child.tag=="ENAMEX":
                                    lab = child.attrib["TYPE"]
                                    l = [f"I-{lab}" for i in t]
                                    l[0] = f"B-{lab}"
                                    l[-1] = f"E-{lab}"
                                    if len(t)==1:
                                        l[0] = f"S-{lab}"
                                else:
                                    l = [default_label for i in t]
                                text += t
                                labels += l
                                t = list(re.sub("[^\S\n]","",child.tail))
                                l = [default_label for i in t]
                                text += t
                                labels += l
                        assert len(text) == len(labels)
                        index = 0
                        for text_line in "".join(text).split("\n"):
                            label = labels[index:index+len(text_line)]
                            write_flag = False
                            for ch,l in zip(text_line,label):
                                write_flag = True
                                save.write(f"{ch} {l}\n")
                            if write_flag:
                                save.write("\n")
                            index = index+len(text_line)+1
                        assert index == len(text)+1,f"{index} {len(text)+1}"
                    line = f.readline()
        
# %%
import json
import os
import time
from tqdm import tqdm, trange

def BMES_to_json(bmes_file, json_file):
    """
    convert bmes format file to json file, json file has two key, including text and label
    Args:
        bmes_file:
        json_file:
    :return:
    """
    texts = []
    with open(bmes_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)
        words = []
        labels = []
        for idx in line_iter:
            line = lines[idx]
            line = line.strip()

            if not line:
                assert len(words) == len(labels), (len(words), len(labels))
                sample = {}
                sample['text'] = words
                sample['label'] = labels
                texts.append(json.dumps(sample, ensure_ascii=False))

                words = []
                labels = []
            else:
                items = line.split()
                words.append(items[0])
                labels.append(items[1])

    with open(json_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write("%s\n"%(text))
# %%
BMES_to_json("temp/ontonotes-name.bmes", "temp/ontonotes-name.json")
# %%
