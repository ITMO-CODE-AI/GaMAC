import numpy as np

from gamac.estimation.internal import Internal
from gamac.pipeline.cvi_predictor import CVIPredictor
from gamac.tests.utils import traverse_data


class TestMeta:
    def test_meta_features(self):
        cvi_predictor = CVIPredictor()
        for data_dict in traverse_data():
            result = cvi_predictor._meta_features(data_dict['data'])
            diff = np.sum(
                np.abs(result - data_dict['features'])
            ).__float__()
            assert diff < 7e-6

    def test_meta_predictions(self, monkeypatch):
        cvi_predictor = CVIPredictor()
        monkeypatch.setattr(cvi_predictor, '_meta_features', lambda df: np.zeros(shape=512))
        for data_dict in traverse_data():
            result = cvi_predictor.run(data_dict['data'])
            assert result in [Internal.OS, Internal.MCR, Internal.BR, Internal.SYM]
