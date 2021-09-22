class IDataLoader:
    def __init__(self, file_name):
        self.read_data_set(file_name)
        self.verify_data()
        self.process_data()

    '''
    读取数据
    '''
    def read_data_set(self, file_name):
        raise Exception("read_data_set is not defined.")
    
    '''
    验证数据
    '''
    def verify_data(self):
        raise Exception("verify_data is not defined.")

    '''
    处理数据
    '''
    def process_data(self):
        raise Exception("process_data is not defined.")

class IModel:
    def __init__(self):
        self.load_model()
    
    '''
    加载外部模型主体, 完成模型配置
    '''
    def load_model(self):
        raise Exception('load_model is not defined.')
    
    '''
    获取最终模型主体
    '''
    def get_model(self):
        raise Exception('get_model is not defined.')

class ITrainer:

    def __init__(self):
        self.dataloader_init()
        self.model_init()
    
    def model_init(self):
        raise Exception('model_init is not defined.')

    def dataloader_init(self):
        raise Exception('dataloader_init is not defined.')

    def train(self):
        raise Exception('train is not defined.')

    def save_model(self):
        raise Exception('save_model is not defined.')

    def eval(self):
        raise Exception('eval is not defined.')

class IPredict:

    def __init__(self):
        self.model_init()
    
    def model_init(self):
        raise Exception('model_init is not defined.')

    def data_process(self):
        raise Exception('data_process is not defined.')

    def pred(self):
        raise Exception('pred is not defined.')

class IAnalysis:

    def __init__(self):
        raise Exception('IAnalysis has no constructor.')