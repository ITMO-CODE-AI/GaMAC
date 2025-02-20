import pickle


class Models:
    def __init__(self, common_root):
        self.common_root = common_root

        self.features_extractor_path = f'{common_root}/feature-extractor.pkl'
        self.meta_classifier_path = f'{common_root}/meta-classifier.pkl'
        self.meta_regressor_path = f'{common_root}/meta-regressor.pkl'

    @staticmethod
    def write_pkl(path, model):
        with open(path, 'wb') as pkl:
            pickle.dump(model, pkl)

    def write_feature_extractor(self, model):
        self.write_pkl(self.features_extractor_path, model)
