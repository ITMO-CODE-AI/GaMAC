from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.spatial.distance import euclidean

from gamac.src.meta.reducers import ReducerProvider
from gamac.src.meta.render import scatter_image
from gamac.src.meta.storage import read_meta_features, traverse_data, create_visual_dir, VISUAL_ROOT


def get_dists(data_dict):
    pair_dists = list()
    for x, y in combinations(data_dict.items(), 2):
        d = euclidean(x[1], y[1])
        pair_dists.append((x[0], y[0], d))
    return sorted(pair_dists, key=lambda p: p[2])


def visualize(data_arr):
    create_visual_dir()
    for reducer in ReducerProvider.get_all():
        reduced = reducer.fit_transform(data_arr)
        if reducer.name == 'pca':
            print(f'EXPLAINED: {reducer.algo.explained_variance_ratio_}')
        xv, yv = reduced[:, 0], reduced[:, 1]
        img_path = f'{VISUAL_ROOT}/{reducer.name}.png'
        scatter_image(xv, yv, colors=None, img_path=img_path)


if __name__ == '__main__':
    meta = traverse_data(read_meta_features)
    dists = get_dists(meta)

    counter = defaultdict(float)
    for x, y, d in dists:
        counter[x] += d
        counter[y] += d

    rates = list(counter.values())
    r_max = np.max(rates)

    sorted_rates = sorted([
        (data_path, value / r_max) for data_path, value in counter.items()
    ], key=lambda p: p[1])
    rating = np.array(sorted_rates)
    print(rating)

    meta_arr = np.array(list(meta.values()))
    visualize(meta_arr)

