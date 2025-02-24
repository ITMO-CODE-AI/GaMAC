import pickle


class Binaries:
    def __init__(self, common_root):
        self.common_root = common_root
        
    @property
    def classifier_extractor_path(self):
        return f'{self.common_root}/classifier-extractor.pkl'
    
    @property
    def classifier_model_path(self):
        return f'{self.common_root}/classifier-model.pkl'

    @property
    def regressor_extractor_path(self):
        return f'{self.common_root}/regressor-extractor.pkl'

    @property
    def regressor_model_path(self):
        return f'{self.common_root}/regressor-model.pkl'

    @staticmethod
    def write_pkl(path, model):
        with open(path, 'wb') as pkl:
            pickle.dump(model, pkl)

    def write_classifier(self, extractor, model):
        self.write_pkl(self.classifier_extractor_path, extractor)
        self.write_pkl(self.classifier_model_path, model)
        
    def write_regressor(self, extractor, model):
        self.write_pkl(self.regressor_extractor_path, extractor)
        self.write_pkl(self.regressor_model_path, model)
