import json

def conll_to_json(src_file, tgt_file, split_tag='\n\n'):
    '''
    CoNLL to Json file.
    src_file: source CoNLL format file.
    tgt_file: output file name.
    split_tag: split tag of each segment.
    Example:
    记 O
    得 O
    小 B_Time
    时 I_Time
    候 I_Time
    
    妈 B_Person
    妈 I_Person
    说 O
    起 O
    哪 O
    个 O
    典 O
    型 O
    败 B_Person
    家 I_Person
    子 I_Person
    形 O
    象 O
    '''
    with open(src_file, encoding="utf-8", mode='r') as f:
        ori_list = f.read().split(split_tag)

    ori_list = [item.split('\n') for item in ori_list]
    ori_list = ori_list[:-1]

    result = []
    for group in ori_list:
        line = {}
        line['text'] = []
        line['label'] = []
        for item in group:
            text, label = item.split(' ')
            line['text'].append(text)
            line['label'].append(label)
        result.append(line)

    with open(tgt_file, encoding='utf-8', mode='w+') as f:
        f.write('')
    with open(tgt_file, encoding='utf-8', mode='a+') as f:
        for line in result:
            f.write('{}\n'.format(json.dumps(line, ensure_ascii=False)))

def cnerta_to_json(src_file, tgt_file, split_tag='\n'):
    '''
    CNERTA to Json file.
    src_file: source CNERTA format file.
    tgt_file: output file name.
    split_tag: split tag of each segment.
    Example:
    {"sentence": "这并不是东方精工第一次终止收购机器人公司", "audio": "BAC009S0706W0325", "entity": [[4, 8, "东方精工", "ORG"]], "speaker_info": "M"}
    '''
    with open(src_file, encoding="utf-8", mode='r') as f:
        ori_list = f.read().split(split_tag)

    ori_list = ori_list[:-1]

    result = []
    for item in ori_list:
        item = json.loads(item)
        line = {}
        line['text'] = [t for t in item['sentence']]
        line['label'] = ['O' for _ in item['sentence']]
        for entity in item['entity']:
            start = entity[0]
            end = entity[1]
            line['label'][start] = 'B-{}'.format(entity[3])
            if end != start + 1:
                line['label'][start + 1: end] = ['I-{}'.format(entity[3]) for _ in range(end - start - 1)]
        result.append(line)

    with open(tgt_file, encoding='utf-8', mode='w+') as f:
        f.write('')
    with open(tgt_file, encoding='utf-8', mode='a+') as f:
        for line in result:
            f.write('{}\n'.format(json.dumps(line, ensure_ascii=False)))

def cluner_to_json(src_file, tgt_file, split_tag='\n'):
    '''
    CNERTA to Json file.
    src_file: source CNERTA format file.
    tgt_file: output file name.
    split_tag: split tag of each segment.
    Example:
    {"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}}
    '''
    with open(src_file, encoding="utf-8", mode='r') as f:
        ori_list = f.read().split(split_tag)

    ori_list = ori_list[:-1]

    result = []
    for item in ori_list:
        item = json.loads(item)
        line = {}
        line['text'] = [t for t in item['text']]
        line['label'] = ['O' for _ in item['text']]
        for label in item['label']:
            for entity in item['label'][label]:
                for entity_range in item['label'][label][entity]:
                    start = entity_range[0]
                    end = entity_range[1] + 1
                    line['label'][start] = 'B-{}'.format(label)
                    if end != start + 1:
                        line['label'][start + 1: end] = ['I-{}'.format(label) for _ in range(end - start - 1)]
        result.append(line)

    with open(tgt_file, encoding='utf-8', mode='w+') as f:
        f.write('')
    with open(tgt_file, encoding='utf-8', mode='a+') as f:
        for line in result:
            f.write('{}\n'.format(json.dumps(line, ensure_ascii=False)))


def generate_tags_from_json(files=[], tgt_file='tags_list.txt'):
    tags = []
    for file in files:
        with open(file, encoding="utf-8", mode='r') as f:
            ori_list = f.read().split('\n')
        ori_list = ori_list[:-1]
        for item in ori_list:
            item = json.loads(item)
            labels = item['label']
            for label in labels:
                if label not in tags:
                    tags.append(label)

    with open(tgt_file, encoding='utf-8', mode='w+') as f:
        f.write('')
    with open(tgt_file, encoding='utf-8', mode='a+') as f:
        for tag in tags:
            f.write('{}\n'.format(tag))
