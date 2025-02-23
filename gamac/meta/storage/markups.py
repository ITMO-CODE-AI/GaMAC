import json
import os
from typing import Tuple, List

AccessorComparison = Tuple[int, int, float]


class AccessorResult:
    def __init__(
            self,
            sorted_indices: List[int],
            all_comparisons: List[AccessorComparison]
    ):
        self.sorted_indices = sorted_indices
        self.all_comparisons = all_comparisons


class Markups:
    def __init__(self, data_root):
        self.data_root = data_root

    def dir_path(self, data_path):
        return f'{self.data_root}/{data_path}/accessors'

    def accessor_path(self, data_path, accessor_id):
        return f'{self.dir_path(data_path)}/{accessor_id}.json'

    def create_markup_dir(self, data_path):
        a_dir = self.dir_path(data_path)
        if not os.path.exists(a_dir):
            os.mkdir(a_dir)

    def accessor_exists(self, data_path, accessor_id):
        a_path = self.accessor_path(data_path, accessor_id)
        return os.path.exists(a_path)

    def write_markup(self, data_path, accessor_id, result: AccessorResult):
        content = {
            'sorted_indices': result.sorted_indices.__str__(),
            'all_comparisons': [
                comparison.__str__() for comparison in result.all_comparisons
            ],
        }
        a_path = self.accessor_path(data_path, accessor_id)
        with open(a_path, 'w') as fp:
            json.dump(content, fp)


    def read_markup(self, data_path, accessor_id) -> AccessorResult:
        a_path = self.accessor_path(data_path, accessor_id)
        with open(a_path, 'r') as fp:
            content = json.load(fp)
        return AccessorResult(
            sorted_indices=eval(content['sorted_indices']),
            all_comparisons=[eval(comp) for comp in content['all_comparisons']]
        )

    def read_markups(self, data_path) -> List[AccessorResult]:
        markups, markup_dir = list(), self.dir_path(data_path)
        for markup_file in os.listdir(markup_dir):
            accessor_id = markup_file.removesuffix(".json")
            markup = self.read_markup(data_path, accessor_id)
            markups.append(markup)
        return markups
