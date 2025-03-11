import json

from sklearn.neighbors import KNeighborsRegressor

from gamac.src.meta.dataset import INTERNAL_MEASURES
from gamac.src.meta.storage import traverse_data, read_meta_features


def build_table(all_data, all_features):
    X, Y = list(), list()
    for data_path, measure_values in all_data.items():
        meta_features = all_features[data_path]
        measure_distances = [
            measure_values[measure_name] for measure_name, _ in INTERNAL_MEASURES
        ]
        X.append(meta_features), Y.append(measure_distances)
    return X, Y


if __name__ == '__main__':
    with open('pre-meta-dataset.json', 'r') as fp:
        all_meta_data = json.load(fp)
    all_meta_features = traverse_data(read_meta_features)

    X, Y = build_table(all_meta_data, all_meta_features)
    regressor = KNeighborsRegressor(n_neighbors=3, weights='distance').fit(X, Y)
    test = read_meta_features('wine-quality-red/tsne')
    result = regressor.predict([test])
    print(result)
