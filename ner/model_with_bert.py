import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ner.crf import CRF


class BiRnnCrf(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_rnn_layers=1, rnn="lstm"):
        super(BiRnnCrf, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(self.embedding_dim, hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, batch_first=True)
        self.crf = CRF(hidden_dim, self.tagset_size)

    def __build_features(self, embeds, masks):

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length.cpu(), batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, embeds, masks, tags):
        features, masks = self.__build_features(embeds, masks)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, embeds, masks):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(embeds, masks)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq