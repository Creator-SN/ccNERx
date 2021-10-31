# %%
import enum
import json
from tqdm import *
from tools.dis_calc import LabelCounter
import matplotlib.pyplot as plt
import os
for index,file in enumerate(sorted(os.listdir("data/weibonew"))):
    if file.endswith(".json"):
        path = os.path.join("data/weibonew",file)
        counter = LabelCounter("data/weibonew/labels.txt")
        print(f"path:{path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="loading train set..."):
                data = json.loads(line.strip())
                text, label = data["text"], data["label"]
                counter.add(label,text)
        counter.sorted_keys()
        labels = list(counter.labels)
        values = [counter.counter[x] for x in labels]
        plt.figure(index+1,figsize=(14,6))
        plt.title(file)
        plt.bar(labels,values,align="center",width=0.4,color="#2ecc71")
        for a,b in zip(labels,values):
            plt.text(a, b+1, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
            
# %%
