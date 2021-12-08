import json
from flask import Flask, request, jsonify
from CC.predicter import NERPredict

app = Flask(__name__)

args = {
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 512,
    'max_scan_num': 1000000,
    'train_file': './data/FN/fj+sc/train(json-400).csv',
    'eval_file': './data/FN/sc-json/dev.csv',
    'test_file': './data/FN/sc-json/test.csv',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/FN/tags_list.txt',
    'loader_name': 'le_loader',
    'output_eval':True,
    "word_embedding_file":"./data/tencent/word_embedding.txt",
    "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "default_tag":"O",
    'batch_size': 4,
    'eval_batch_size': 64,
    'do_shuffle': True,
    "use_gpu": True,
    "debug": True,
    'model_name': 'LEBert',
    'task_name': 'FN-fj+sc(400)-sc-LeBert',
    'lstm_crf_model_file':'./save_model/FN-fj+sc(400)-sc-LeBert/lstm_crf/lstm_crf_6000.pth',
    'bert_model_file':'./save_model/FN-fj+sc(400)-sc-LeBert/LEBert/LEBert_6000.pth',
}

predict = NERPredict(**args)

# predict(["BMI:23.18kg/㎡体重正常","胆囊胆固醇沉着","胆囊结石", "心肌损伤酶谱：正常;右肾囊肿"])

@app.route('/')
def default():
    return 'ACM NER'


@app.route('/pred', methods=['post'])
def pred():
    if not request.form.getlist('text'):  # 检测是否有数据
        return ('No Data Found.')
    data = request.form.getlist('text')
    outputs = predict(data)[0:]
    # text = ''.join(text)
    entities_list = []
    text_list = []
    labels_labels = []
    for i in range(len(outputs)):
        text,labels = outputs[i]
        entities = []
        e = ''
        for j in range(len(labels)):
            # t = text[j]
            if labels[j] == 'O' and e != '':
                entities.append(e.replace('#', ''))
                e = ''
            elif labels[j] == 'B-KEYWORD' or labels[j] == 'I-KEYWORD':
                e += text[j]
        if e != '':
            entities.append(e.replace('#', ''))
        entities_list.append(entities)
        text_list.append(text)
        labels_labels.append(labels)
    
    # 过滤错误匹配项
    with open('./data/filter_list', encoding='utf-8', mode='r') as f:
        filter_list = f.read().split('\n')
        if filter_list[-1] == '':
            filter_list = filter_list[:-1]
    for entities in entities_list:
        l = len(entities)
        for i in range(l - 1, -1, -1):
            for filter in filter_list:
                if not entities[i].find(filter) < 0:
                    del entities[i]
                    break
    
    result = {}
    result['text'] = text_list
    result['labels'] = labels_labels
    result['entities'] = entities_list
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    app.run()


