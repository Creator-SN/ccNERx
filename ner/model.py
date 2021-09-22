import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel
from ner.crf import CRF
from ner.model_with_bert import BiRnnCrf
from ICCSupervised.ICCSupervised import IModel

class BertNER(IModel):

    def __init__(self, bert_config_file_name, pretrained_file_name, vocab_size, tagset_size, hidden_dim):
        self.load_model(bert_config_file_name, pretrained_file_name, vocab_size, tagset_size, hidden_dim)
    
    def load_model(self, bert_config_file_name, pretrained_file_name, vocab_size, tagset_size, hidden_dim):
        config = BertConfig.from_json_file(bert_config_file_name)
        self.model = BertModel.from_pretrained(pretrained_file_name, config=config)
        self.birnncrf = BiRnnCrf(vocab_size=vocab_size, tagset_size=tagset_size, embedding_dim=config.hidden_size, hidden_dim=hidden_dim)
    
    def get_model(self):
        return self.model, self.birnncrf
    
    def __call__(self):
        return self.get_model()