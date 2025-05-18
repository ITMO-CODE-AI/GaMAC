"""
Interface to handle fitted models storage
"""

import pickle


class Binaries:
    def __init__(self, common_root):
        self.common_root = common_root
        
    @property
    def extractor_path(self):
        return f'{self.common_root}/extractor.pkl'
    
    @property
    def classifier_path(self):
        return f'{self.common_root}/classifier.pkl'

    @property
    def regressor_path(self):
        return f'{self.common_root}/regressor.pkl'

    @staticmethod
    def write_pkl(path, model):
        with open(path, 'wb') as pkl:
            pickle.dump(model, pkl)

    def write_extractor(self, extractor):
        self.write_pkl(self.extractor_path, extractor)

    def write_classifier(self, classifier):
        self.write_pkl(self.classifier_path, classifier)
        
    def write_regressor(self, regressor):
        self.write_pkl(self.regressor_path, regressor)
