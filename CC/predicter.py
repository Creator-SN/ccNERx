import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertTokenizer, BertModel
from ICCSupervised.ICCSupervised import IPredict
from CC.loaders.cn_loader import BERTDataManager
from CC.loaders import *
from CC.loaders.utils import *
from CC.model import CCNERModel


class NERPredict(IPredict):

    def __init__(self, use_gpu,
                 bert_config_file_name,
                 vocab_file_name,
                 tags_file_name,
                 bert_model_path,
                 lstm_crf_model_path,
                 hidden_dim,
                 padding_length=512, **args):
        parser = KwargsParser(debug=True) \
            .add_argument("use_gpu", bool, defaultValue=False) \
            .add_argument("loader_name", str, defaultValue="le_loader") \
            .add_argument("model_name", str, defaultValue="LEBert") \
            .add_argument("lstm_crf_model_file", str) \
            .add_argument("bert_model_file", str) \
            .add_argument("hidden_dim", int) \
            .add_argument("bert_vocab_file", str) \
            .add_argument("bert_config_file_name", str) \
            .add_argument("tag_file", str) \
            .add_argument("padding_length", int, 512) \
            .parse(self, **args)

        if self.loader_name == "le_loader":
            self.loader = LLoader(**args)
        elif self.loader_name == "cn_loader":
            self.data_manager_init(self.bert_vocab_file, self.tag_file)
            self.tokenizer = BertTokenizer.from_pretrained(self.vocab_file)
        self.model_init()

    def data_manager_init(self, vocab_file_name, tags_file_name):
        tags_list = BERTDataManager.ReadTagsList(tags_file_name)
        tags_list = [tags_list]
        self.dm = BERTDataManager(tags_list=tags_list,
                                  vocab_file_name=vocab_file_name)

    def model_init(self):
        model_args = {
            'model_name': self.model_name,
            'bert_config_file_name': self.bert_config_file_name,
            'tagset_size': self.tag_size,
            'hidden_dim': self.hidden_dim
        }
        # TODO: Add self attributes
        if 'word_embedding_file' in args:
            model_args['pretrained_embeddings'] = self.vocab_embedding
        if 'pretrained_file_name' in args:
            model_args['pretrained_file_name'] = args['pretrained_file_name']

        self.bert_ner = CCNERModel(**model_args)
        self.model, self.birnncrf = self.bert_ner()

    def data_process(self, sentences):
        result = []
        pad_tag = '[PAD]'
        if type(sentences) == str:
            sentences = [sentences]
        max_len = 0
        for sentence in sentences:
            encode = self.tokenizer.encode(
                sentence, add_special_tokens=True, max_length=self.padding_length, truncation=True)
            result.append(encode)
            if max_len < len(encode):
                max_len = len(encode)

        for i, sentence in enumerate(result):
            remain = max_len - len(sentence)
            for _ in range(remain):
                result[i].append(self.dm.wordToIdx(pad_tag))
        return torch.tensor(result)

    def pred(self, sentences):
        sentences = self.data_process(sentences)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.birnncrf.to(device)

        with torch.no_grad():
            if torch.cuda.is_available():
                self.model.cuda()
                self.birnncrf.cuda()
                sentences = sentences.cuda()

            outputs = self.model(input_ids=sentences,
                                 attention_mask=sentences.gt(0))
            hidden_states = outputs[0]
            scores, tags = self.birnncrf(hidden_states, sentences.gt(0))
        final_tags = []
        decode_sentences = []

        for item in tags:
            final_tags.append([self.dm.idx_to_tag[tag] for tag in item])

        for item in sentences.tolist():
            s = []
            for word_idx in item:
                s.append(self.dm.idxToWord(word_idx))
            decode_sentences.append(s)

        return (scores, tags, final_tags, decode_sentences)

    def __call__(self, sentences):
        return self.pred(sentences)
