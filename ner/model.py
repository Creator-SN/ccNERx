import torch
import torch.nn as nn
from transformers import BertConfig
from ner.LBert import WCBertModel, BertPreTrainedModel
from ner.crf import CRF
from ner.model_with_bert import BiRnnCrf
from ICCSupervised.ICCSupervised import IModel


class BertNER(IModel):

    def __init__(self, bert_config_file_name, pretrained_file_name, pretrained_embeddings, tagset_size, hidden_dim):
        self.load_model(bert_config_file_name, pretrained_file_name,
                        pretrained_embeddings, tagset_size, hidden_dim)

    def load_model(self, bert_config_file_name, pretrained_file_name, pretrained_embeddings, tagset_size, hidden_dim):
        config = BertConfig.from_json_file(bert_config_file_name)
        self.model = LBertModel.from_pretrained(
            pretrained_file_name, pretrained_embeddings=pretrained_embeddings, config=config)
        self.birnncrf = BiRnnCrf(
            tagset_size=tagset_size, embedding_dim=config.hidden_size, hidden_dim=hidden_dim)

    def get_model(self):
        return self.model, self.birnncrf

    def __call__(self):
        return self.get_model()


class LBertModel(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape还没去看
    '''

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.bert = WCBertModel(config)
        self.dropout = nn.Dropout(config.HP_dropout)

        self.init_weights()

        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):
        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=args['matched_word_embeddings'],
            matched_word_mask=args['matched_word_mask']
        )

        return outputs
