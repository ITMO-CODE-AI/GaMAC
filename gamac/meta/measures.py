from collections import defaultdict

from gamac.meta.impl.cvi import *
from gamac.meta.storage import CONTENTS, COMPUTED, GATHERED

def compute_measures(data_path):
    print(f"*** STARTED {data_path} ***")
    evaluations = defaultdict(list)
    eval_measures = INTERNAL_MEASURES

    if COMPUTED.measures_exist(data_path):
        calculated = COMPUTED.read_measures_val(data_path)

        eval_measures = [item for item in INTERNAL_MEASURES if item[0] not in calculated]
        remain_measures = {item[0] for item in INTERNAL_MEASURES if item[0] in calculated}

        for calc_measure, orderings in calculated.items():
            if calc_measure in remain_measures:
                evaluations[calc_measure] = orderings

    if len(eval_measures) == 0:
        print(f"*** SKIPPED {data_path} ***")
        return
    data = CONTENTS.read_gen_data(data_path)
    partitions = CONTENTS.read_partitions(data_path)

    for idx, partition in enumerate(partitions):
        print(f"+++ PARTITION {idx} +++")
        clusters = Clusters.create(data, partition)
        for measure_name, measure_fun in eval_measures:
            print(f"### {measure_name} ###")
            measure_value = measure_fun(clusters)
            evaluations[measure_name].append(measure_value)
    COMPUTED.write_measures(data_path, evaluations)


def traverse_measures():
    pre_values = GATHERED.read_pre_values()
    filtered = [data_path for data_path, pre_val in pre_values.items() if pre_val > 1]
    for data_path in sorted(filtered):
        compute_measures(data_path)


if __name__ == '__main__':
    traverse_measures()