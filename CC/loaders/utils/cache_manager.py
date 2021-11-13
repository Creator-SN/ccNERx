import os
import pickle
from typing import Any


class FileCache():

    def __init__(self,dir="./temp") -> None:
        if os.path.exists(dir):
            os.makedirs(dir)
        self.root = dir

    def exists(self,key:str)->bool:
        path = self._get_path(key)
        return os.path.exists(path)

    def save(self,key:str,obj:Any,overwrite:bool=False)->bool:
        path = self._get_path(key)
        if self.exists(key) and not overwrite:
            return False
        with open(path,"wb") as f:
            pickle.dump(obj,f)
        return True

    def load(self,key)->Any:
        path = self._get_path(key)
        if self.exists(path):
            with open(path,"rb") as f:
                return pickle.load(f)
        return None
        
    def _get_path(self,key:str)->str:
        return os.path.join(self.root,key)
