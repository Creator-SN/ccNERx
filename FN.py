# %%
from CC.trainer import NERTrainer

# %%
trainer = NERTrainer(10, [0, 1, 2, 3],
                     bert_config_file_name='./model/chinese_wwm_ext/bert_config.json',
                     pretrained_file_name='./model/chinese_wwm_ext/pytorch_model.bin',
                     hidden_dim=150,
                     train_file_name='./data/FN_v2/train.csv',
                     vocab_file_name='./model/chinese_wwm_ext/vocab.txt',
                     tags_file_name='./data/FN_v2/tags_list.txt',
                     eval_file_name='./data/FN_v2/dev.csv',
                     word_tag_split='\t',
                     pattern='(\tO',
                     batch_size=300,
                     eval_batch_size=300,
                     task_name='FN_v2')

for i in trainer():
    a = i

# %%
from CC.predicter import NERPredict

# %%
predict = NERPredict(True,
                     bert_config_file_name='./model/chinese_wwm_ext/bert_config.json',
                     vocab_file_name='./model/chinese_wwm_ext/vocab.txt',
                     tags_file_name='./data/FN_v2/tags_list.txt',
                     bert_model_path='./save_model/bert/5716df27_bert.pth',
                     lstm_crf_model_path='./save_model/lstm_crf/5716df27_lstm_crf.pth',
                     hidden_dim=150)

# %%
labels, text = predict(["1.心电图室:(1)窦性心律(2)正常心电图2.放射科(数字化X线):胸部正侧位片:心肺未见明显异常　3.B超室:肝、胆、胰、脾、左肾、右肾、左输尿管、右输尿管、膀胱、子宫、左附件区、右附件区未见明显占位性病变4.彩超室:(1)甲状腺左叶低回声结节,考虑结节性甲状腺肿,建议随访复查(2)左侧乳腺低回声结节,建议随访复查；双侧乳腺增生5.妇科:(1)宫颈:肥大、潴留囊肿(2)外阴:正常(3)阴道:正常(4)分泌物:未见异常(5)子宫:正常(6)附件:未触及明显占位性病变6.H-TCT:(1)细胞病理学诊断:无上皮病变或恶性病变（NILM）(2)备注及建议:请在一年内复查7.检验科:(1)白带常规: 清洁度: III度(2)血常规五分类: 平均血小板体积: 11.9↑(3)生化全套2: 谷氨酰转肽酶: 64↑(4)血同型半胱氨酸: 正常(5)甲胎蛋白定量: 正常(6)癌胚抗原定量: 正常(7)CA242: 正常(8)NSE: 正常(9)CYFRA21-1: 正常(10)尿常规: 正常", "1.一般项目:(1)收缩压:98mmHg(2)舒张压:60mmHg(3)血压结论:正常血压(4)脉搏:70次/分(5)身高:167cm(6)体重:84.4kg(7)体重指数:30.26,肥胖2.心电图室:(1)窦性心律(2)T波改变3.放射科(数字化X线):考虑左肺少许硬结灶及纤维化,请结合临床4.放射科(MR):1.颈椎、胸椎、腰椎退行性变。2.C4/5、C5/6椎间盘膨隆。3.L4/5椎间盘突出（偏左）,L5/S1椎间盘突出（中央型）。5.彩超室:(1)甲状腺多发结节,考虑结节性甲状腺肿,建议定期复查(2)肝右叶囊肿；中度脂肪肝(3)前列腺增大(4)胆、胰、脾、左肾、右肾、左输尿管、右输尿管、膀胱未见明显占位性病变6.检验科:(1)血常规五分类: 正常(2)生化全套4: 谷丙转氨酶: 59↑; 甘油三脂: 2.12↑(3)糖化血红蛋白: 正常(4)男性肿瘤标志物5: 正常(5)乙肝两对半定量: 乙肝表面抗体定量: 197.82↑; 乙肝核心抗体定量: 2.73↑(6)胃蛋白酶原: 正常(7)食物不耐受检测7项(新): 正常(8)尿常规+尿沉渣: 正常"])[2:]

# %%
for idx, label in enumerate(labels):
    t = text[idx]
    for j, item in enumerate(label):
        print('{}\t{}'.format(t[j], item))

# %%
import re
from tqdm import tqdm

with open('./data/FN_v2/0323_top100.csv', mode='r') as f:
    ori_list = f.read().split('\n')
if ori_list[-1].strip() == '':
    ori_list = ori_list[:-1]

result_list = []
for line in tqdm(ori_list):
    result = ''
    line = line.split('\t')[2].strip()
    line = line.replace('\\r', '\r')
    seg_list = re.split('。|；', line)
    labels, text = predict(seg_list)[2:]
    word = ''
    for idx, _ in enumerate(labels):
        for i, __ in enumerate(labels[idx]):
            if labels[idx][i] == 'B-KEYWORD':
                word = text[idx][i]
            elif labels[idx][i] == 'I-KEYWORD':
                word += text[idx][i]
            else:
                if word != '':
                    result = '{};{}'.format(result, word) if result != '' else word
                    word = ''
    result_list.append(result)

with open('./data/FN_v2/0323_top100_keywords.csv', mode='w+') as f:
    f.write('')

with open('./data/FN_v2/0323_top100_keywords.csv', mode='a+') as f:
    for item in result_list:
        f.write('{}\n'.format(item))

# %%
