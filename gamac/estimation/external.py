from enum import Enum

from cupy.typing import NDArray

from gamac.estimation.functions import f1


class External(Enum):
    F1 = f1


class ExternalEvaluator:
    def __init__(
            self,
            classes: NDArray,
            classes_k: int,
            labels: NDArray,
            labels_k: int,
    ):
        self.classes, self.classes_k = classes, classes_k
        self.labels, self.labels_k = labels, labels_k


    def evaluate(self, measure: External) -> float:
        return measure.value(
            self.classes, self.classes_k, self.labels, self.labels_k
        )