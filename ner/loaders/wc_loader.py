class LLoader(IDataLoader):
    def __init__(self, **args):
        '''
            Args:
                word_embedding_file: word embedding file path
                vocab_file: vocab file path

        '''
        assert args["word_embedding_file"] != None
        assert args["vocab_file"] != None
        self.read_data_set(args["train_file_name"],
                           args["eval_file_name"], args["test_file_name"])

    def read_data_set(self, train_file_name: str, eval_file_name: str = None, test_file_name: str = None):
        pass

    def verify_data(self):
        raise Exception("verify_data is not defined.")

    def process_data(self):
        raise Exception("process_data is not defined.")
