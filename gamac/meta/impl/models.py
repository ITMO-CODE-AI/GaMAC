from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from umap import UMAP
from xgboost import XGBClassifier, XGBRegressor

from gamac.meta.impl.utils import Reducer, Producer
from gamac.meta.premeta import NUM_MEASURES


class ModelProvider:
    @staticmethod
    def get_classifiers():
        return [
            *ModelProvider._knn_classifiers(),
            *ModelProvider._xgb_classifiers()
        ]

    @staticmethod
    def get_best_classifier_extractor():
        return UMAP(
            n_components=12,
            n_neighbors=NUM_MEASURES,
            min_dist=0.07,
            low_memory=False,
            random_state=42,
        )

    @staticmethod
    def get_best_xgb_classifier():
        # f1 0.9126003519147281
        return XGBClassifier(
            n_estimators=80,
            learning_rate=0.2,
            max_depth=7,
            objective='multi:softmax',
            num_class=4,
            tree_method='exact',
            seed=42
        )

    @staticmethod
    def get_best_knn_classifier():
        # f1 0.9062663323163725
        return KNeighborsClassifier(
            n_neighbors=4,
            weights=ModelProvider._knn_weighted_dist
        )

    @staticmethod
    def _knn_weighted_dist(x):
        return 1 / (1 + x)

    @staticmethod
    def _knn_classifiers():
        classifiers = list()
        for n_neighbours in [3, 4, 5, 6, 7, 8]:
            classifiers.append(
                Producer(
                    KNeighborsClassifier(
                        n_neighbors=n_neighbours,
                        weights=ModelProvider._knn_weighted_dist
                    ),
                    f'knn-{n_neighbours}'
                )
            )
        return classifiers

    @staticmethod
    def _xgb_classifiers():
        classifiers = list()
        for n_estimators in [50, 80, 120]:
            for lr in [0.05, 0.2]:
                for max_depth in [3, 5, 7]:
                    classifiers.append(
                        Producer(
                            XGBClassifier(
                                n_estimators=n_estimators,
                                learning_rate=lr,
                                max_depth=max_depth,
                                objective='multi:softmax',
                                num_class=4,
                                tree_method='exact',
                                seed=42
                            ),
                            f'xgb-{n_estimators}-{lr}-{max_depth}'
                        )
                    )
        return classifiers

    @staticmethod
    def get_regressors():
        return [
            *ModelProvider._knn_regressors(),
            *ModelProvider._xgb_regressors()
        ]

    @staticmethod
    def get_best_regressor_extractor():
        return UMAP(
            n_components=15,
            n_neighbors=NUM_MEASURES,
            min_dist=0.07,
            low_memory=False,
            random_state=42,
        )

    @staticmethod
    def get_best_xgb_regressor():
        # smape -0.09655866014085755
        return XGBRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            objective='reg:squarederror',
            num_target=4,
            multi_strategy='one_output_per_tree',
            tree_method='exact',
            seed=42,
        )

    @staticmethod
    def get_best_knn_regressor():
        # smape -0.09937646604388396
        return KNeighborsRegressor(
            n_neighbors=8,
            weights=ModelProvider._knn_weighted_dist
        )

    @staticmethod
    def _knn_regressors():
        regressors = list()
        for n_neighbours in [3, 4, 5, 6, 7, 8]:
            regressors.append(
                Producer(
                    KNeighborsRegressor(
                        n_neighbors=n_neighbours,
                        weights=ModelProvider._knn_weighted_dist
                    ),
                    f'knn-{n_neighbours}'
                )
            )
        return regressors

    @staticmethod
    def _xgb_regressors():
        regressors = list()
        for n_estimators in [50, 80, 120]:
            for lr in [0.05, 0.2]:
                for max_depth in [3, 5, 7]:
                    regressors.append(
                        Producer(
                            XGBRegressor(
                                n_estimators=n_estimators,
                                learning_rate=lr,
                                max_depth=max_depth,
                                objective='reg:squarederror',
                                num_target=4,
                                multi_strategy='one_output_per_tree',
                                tree_method='exact',
                                seed=42,
                            ),
                            f'xgb-{n_estimators}-{lr}-{max_depth}'
                        )
                    )
        return regressors

    @staticmethod
    def get_extractors():
        return [
            *ModelProvider._umap_extractor(),
        ]

    @staticmethod
    def _umap_extractor():
        reducers = list()
        for n_components in [5, 7, 10, 12, 15]:
            for min_dist in [0.03, 0.07, 0.12]:
                for n_neighbours in [NUM_MEASURES, 2 * NUM_MEASURES, 3 * NUM_MEASURES]:
                    reducers.append(
                        Reducer(
                            f'umap-{n_components}-{min_dist}-{n_neighbours}',
                            UMAP(
                                n_components=n_components,
                                n_neighbors=n_neighbours,
                                min_dist=min_dist,
                                low_memory=False,
                                random_state=42,
                            ),
                        )
                    )
        return reducers
