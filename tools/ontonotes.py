import os
import json

def ontonote_to_bmes(ontonotes_dir:str):
    # scan map.txt
    file_list = os.listdir(ontonotes_dir)
    data = []
    if "map.txt" in file_list:
        map_txt  = os.path.join(ontonotes_dir,"map.txt")
        
    else:
        for file in file_list:
            path=os.path.join(ontonotes_dir,file) 
            if os.path.isdir(path):
                data+=ontonote_to_bmes(path)
        