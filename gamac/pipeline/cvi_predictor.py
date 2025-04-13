"""Основной скрипт этапа подсказки мер"""
from gamac.data.data_pipeline import DataFrameType
from gamac.estimation.internal import Internal


class CVIPredictor:
    """summary"""

    @staticmethod
    def run(df: DataFrameType) -> Internal:
        return Internal.C_INDEX
