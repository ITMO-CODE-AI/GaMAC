"""
Compute meta-features on available datasets
"""

from gamac.meta.impl.mfs import compute_mfs
from gamac.meta.storage import COMPUTED, CONTENTS, traverse_data


def compute_meta_features(data_path):
    print(f"--- STARTED {data_path} ---")
    if COMPUTED.meta_features_exist(data_path):
        print(f"--- SKIPPED {data_path} ---")
        return
    data = CONTENTS.read_gen_data(data_path)
    meta_features = compute_mfs(data)
    COMPUTED.write_meta_features(data_path, meta_features)


if __name__ == "__main__":
    traverse_data(compute_meta_features)