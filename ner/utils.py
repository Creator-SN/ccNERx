import re
import random

class DataManager():

    def __init__(self, tags_list, vocab_file_name=None, sentences=None, pad_tag='[PAD]'):
        if vocab_file_name != None:
            self.word_to_idx, self.idx_to_word = self.WordIdxFromVocabInit(vocab_file_name)
        else:
            self.word_to_idx, self.idx_to_word = self.WordIdxInit(sentences, pad_tag)
        self.tag_to_idx, self.idx_to_tag = self.TagIdxInit(tags_list, pad_tag)
    
    def wordToIdx(self, word):
        try:
            return self.word_to_idx[word]
        except:
            return self.word_to_idx['[UNK]']
    
    def idxToWord(self, idx):
        try:
            return self.idx_to_word[idx]
        except:
            return '[UNK]'
    
    def tagToIdx(self, tag):
        try:
            return self.tag_to_idx[tag]
        except:
            return 0
    
    def encode(self, sentence, tags, pad_tag='[PAD]', padding_length=100):
        sentence, tags = self.padding_train(sentence, tags, pad_tag, padding_length)
        sentence = [self.wordToIdx(word) for word in sentence]
        tags = [self.tagToIdx(tag) for tag in tags]
        return sentence, tags
    
    def decode(self, sentence, tags, pad_tag='[PAD]'):
        sentence = [self.idx_to_word[idx] for idx in sentence]
        tags = [self.idx_to_tag[idx] for idx in tags]
        count = 0
        for i in tags:
            count += 1
            if i == pad_tag:
                break
        return sentence[:count], tags[:count]
    
    @staticmethod
    def generate_vocab(sentences_list, save_path, pad_tag='[PAD]'):
        result = ""
        word_to_idx, idx_to_word = DataManager.WordIdxInit(sentences_list, pad_tag)
        for word in word_to_idx:
            result += '{}\n'.format(word)
        result += '[UNK]\n'
        with open(save_path, mode='w+', encoding='utf-8') as f:
            f.write(result)

    @staticmethod
    def WordIdxInit(sentences_list, pad_tag='[PAD]'):
        count = 1
        word_to_idx = {}
        idx_to_word = {}
        word_to_idx[pad_tag] = 0
        idx_to_word[0] = pad_tag
        for item in sentences_list:
            for word in item:
                if word not in word_to_idx:
                    word_to_idx[word] = count
                    idx_to_word[count] = word
                    count += 1
        return word_to_idx, idx_to_word
    
    @staticmethod
    def WordIdxFromVocabInit(vocab_file_name):
        with open(vocab_file_name, encoding='utf-8') as f:
            vocab_list = f.read().split('\n')
        word_to_idx = {}
        idx_to_word = {}
        for idx, word in enumerate(vocab_list):
            word_to_idx[word] = idx
            idx_to_word[idx] = word
        return word_to_idx, idx_to_word

    @staticmethod
    def TagIdxInit(tags_list, pad_tag='[PAD]'):
        count = 1
        tag_to_idx = {}
        idx_to_tag = {}
        tag_to_idx[pad_tag] = 0
        idx_to_tag[0] = pad_tag
        for item in tags_list:
            for tag in item:
                if tag not in tag_to_idx:
                    tag_to_idx[tag] = count
                    idx_to_tag[count] = tag
                    count += 1
        return tag_to_idx, idx_to_tag

    '''
    读取数据
    file_name: 文件路径
    word_tag_split: 单词标签间隔符, 默认是空格
    '''
    @staticmethod
    def ReadData(file_name, word_tag_split=' '):
        sentences = []
        tags_list = []
        with open(file_name, encoding='utf-8') as f:
            oriList = f.read().split('\n')
        sentence = []
        tags = []
        for line in oriList:
            line = line.strip()
            arr = line.split(word_tag_split)
            if len(arr) < 2:
                sentences.append(sentence)
                tags_list.append(tags)
                sentence = []
                tags = []
            else:
                sentence.append(arr[0])
                tags.append(arr[1])
        return sentences, tags_list
    
    '''
    细粒度读取数据(分割更多的句子)
    '''
    @staticmethod
    def ReadDataExtremely(file_name, word_tag_split=' ', pattern='， O'):
        sentences = []
        tags_list = []
        with open(file_name, encoding='utf-8') as f:
            txt = f.read().replace(pattern, '')
            oriList = txt.split('\n')
        sentence = []
        tags = []
        for line in oriList:
            line = line.strip()
            arr = line.split(word_tag_split)
            if len(arr) < 2:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    tags_list.append(tags)
                sentence = []
                tags = []
            else:
                sentence.append(arr[0])
                tags.append(arr[1])
        return sentences, tags_list
    
    '''
    合并读取(用于初始化生成单词表)
    '''
    @staticmethod
    def ReadMultiData(file_names=[], word_tag_split=' '):
        sentences = []
        tags_list = []
        for file_name in file_names:
            s, t = DataManager.ReadData(file_name, word_tag_split)
            sentences += s
            tags_list += t
        return sentences, tags_list
    
    '''
    读取标签
    '''
    @staticmethod
    def ReadTagsList(file_name):
        with open(file_name, encoding='utf-8') as f:
            tags_list = f.read().split('\n')
        return tags_list
    
    '''
    分割数据
    '''
    @staticmethod
    def DataSplit(sentences, tags_list, ratio=0.8):
        length = len(sentences)
        ran_index = [i for i in range(length)]
        random.shuffle(ran_index)
        ratio = int(length * ratio)
        train_index, val_index = ran_index[:ratio], ran_index[ratio:length]
        t_s = [sentences[i] for i in train_index]
        t_t = [tags_list[i] for i in train_index]
        v_s = [sentences[i] for i in val_index]
        v_t = [tags_list[i] for i in val_index]
        return t_s, t_t, v_s, v_t
    
    '''
    填充一条训练集
    sentence: 输入句子['a', 'b', 'c']
    tags: 输入标签['o', 'o', 'o']
    pad_tag: 填充标签
    padding_length: 最大填充长度
    '''
    @staticmethod
    def padding_train(sentence, tags, pad_tag='[PAD]', padding_length=100):
        if len(sentence) < padding_length:
            remain = padding_length - len(sentence)
            for i in range(remain):
                sentence.append(pad_tag)
                tags.append(pad_tag)
        else:
            sentence = sentence[:padding_length]
            tags = tags[:padding_length]
        return sentence, tags
