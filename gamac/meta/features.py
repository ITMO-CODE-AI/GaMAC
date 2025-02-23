from gamac.meta.impl.mfs import compute_mfs
from gamac.meta.storage import COMPUTED, CONTENTS, GATHERED


def compute_meta_features(data_path):
    print(f"--- STARTED {data_path} ---")
    if COMPUTED.meta_features_exist(data_path):
        print(f"--- SKIPPED {data_path} ---")
        return
    data = CONTENTS.read_gen_data(data_path)
    meta_features = compute_mfs(data)
    COMPUTED.write_meta_features(data_path, meta_features)


def traverse_features():
    pre_values = GATHERED.read_pre_values()
    filtered = [data_path for data_path, pre_val in pre_values().items() if pre_val > 1]
    for data_path in sorted(filtered):
        compute_meta_features(data_path)


if __name__ == "__main__":
    traverse_features()