import os

from gamac.meta.storage.computed import Computed
from gamac.meta.storage.contents import Contents
from gamac.meta.storage.models import Models
from gamac.meta.storage.gathered import Gathered
from gamac.meta.storage.markups import Markups, AccessorResult

DATA_ROOT = "data"
COMMON_ROOT = "common"

MARKUPS = Markups(data_root=DATA_ROOT)
CONTENTS = Contents(data_root=DATA_ROOT)
COMPUTED = Computed(data_root=DATA_ROOT)
GATHERED = Gathered(common_root=COMMON_ROOT)
MODELS = Models(common_root=COMMON_ROOT)


def traverse_data(handler):
    result, datasets = dict(), os.listdir(DATA_ROOT)
    for dataset in sorted(datasets):
        contents = os.listdir(f'{DATA_ROOT}/{dataset}')
        for subdir in sorted(contents):
            if not os.path.isdir(subdir):
                continue
            data_path = f'{dataset}/{subdir}'
            try:
                result[data_path] = handler(data_path)
                print(f'HANDLED {data_path}')
            except FileNotFoundError:
                print(f'SKIPPED {data_path}')
    return result
