import pytest

from gamac.estimation.internal import InternalEvaluator, Internal
from gamac.tests.utils import traverse_data


class TestMeasures:
    @pytest.mark.parametrize(
        "measure_name,delta",
        [
            ('BR', 7e-9),
            ('OS', 7e-2),
            ('MCR', 7e-4),
            ('SYM', 7e-6),
    ])
    def test_internal(self, measure_name, delta):
        measure = Internal[measure_name]
        for data_dict in traverse_data():
            expected = data_dict['measures'][measure_name]
            fake_pivots = {measure: 0.0}
            evaluator = InternalEvaluator(data_dict['data'], fake_pivots)
            for idx, partition in enumerate(data_dict['partitions']):
                result = evaluator.evaluate(partition)[measure]
                diff = abs(expected[idx] - result)
                assert diff < delta
