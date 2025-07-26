from enum import Enum

from gamac.estimation.functions import f1_micro, f1_macro


class External(Enum):
    F1_MICRO = (f1_micro,)
    F1_MACRO = (f1_macro,)