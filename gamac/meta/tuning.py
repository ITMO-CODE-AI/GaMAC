import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from gamac.meta.build import get_smooth_labels, shuffle_data, collect_raw_meta_dataset
from gamac.meta.impl.models import ModelProvider


def tune_classifier(x_orig, y_orig, weights):
    classifiers_stat = list()
    for extractor in ModelProvider.get_extractors():
        ext_model = extractor.algo.fit(x_orig)
        x_reduced = ext_model.transform(x_orig)
        z_smooth = get_smooth_labels(x_reduced, y_orig, weights)
        x, z = shuffle_data(x_reduced, z_smooth)
        for classifier in ModelProvider.get_classifiers():
            classifier_dict = cross_validate(
                classifier.algo, x, z, cv=20, scoring='f1_macro'
            )
            score = np.mean(classifier_dict['test_score'])
            result = (extractor.name, classifier.name, score)
            print(result)
            classifiers_stat.append(result)
    return classifiers_stat


def smape_loss(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    norm = np.abs(y_true) + np.abs(y_pred)
    return (diff / norm).mean()

def tune_regressors(x_orig, y_orig):
    smape_score = make_scorer(smape_loss, greater_is_better=False)
    regressors_stat = list()
    for extractor in ModelProvider.get_extractors():
        ext_model = extractor.algo.fit(x_orig)
        x_reduced = ext_model.transform(x_orig)
        x, y = shuffle_data(x_reduced, y_orig)
        for regressor in ModelProvider.get_regressors():
            regressor_dict = cross_validate(
                regressor.algo, x, y, cv=20, scoring=smape_score
            )
            score = np.mean(regressor_dict['test_score'])
            result = (extractor.name, regressor.name, score)
            print(result)
            regressors_stat.append(result)
    return regressors_stat


def tune():
    _, x_orig, y_orig, weights = collect_raw_meta_dataset()

    print("=== CLASSIFIERS ===")
    tune_classifier(x_orig, y_orig, weights)

    print("=== REGRESSORS ===")
    tune_regressors(x_orig, y_orig)

if __name__ == '__main__':
    tune()
