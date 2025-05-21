import pytest
import pandas as pd
import numpy as np
import os
from tempfile import NamedTemporaryFile
from datetime import datetime

from sklearn.preprocessing import StandardScaler

from gamac.data.table_preprocessing import table_preprocessing

@pytest.fixture
def sample_data():
    """Фикстура с примером данных для тестирования предобработки."""
    data = {
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10.5, 20.3, np.nan, 40.1, 50.0],
        'categorical1': ['a', 'b', 'a', 'c', 'b'],
        'categorical2': ['x', 'y', 'y', 'x', np.nan],
        'target1': [0, 1, 0, 1, 0],
        'target2': [1, 0, 1, 0, 1],
        'date_col': pd.date_range('2020-01-01', periods=5),
        'bool_col': [True, False, True, False, True],
        'high_unique': [0.1, 0.2, 0.3, 0.4, 0.5],
        'low_unique': [1, 1, 1, 2, 2]
    }
    return pd.DataFrame(data)


def test_infer_columns(sample_data):
    """Тест автоматического определения типов столбцов."""
    processed = table_preprocessing(
        sample_data,
        unknown_column_action='infer',
        verbose=False
    )
    assert processed.shape == (5, 13)


def test_nan_drop_row(sample_data):
    """Тест удаления строк с NaN."""
    processed = table_preprocessing(
        sample_data,
        nan_action='drop row',
        verbose=False
    )
    assert processed.shape[0] == 3


def test_scaling_methods():
    """Тест методов масштабирования."""
    df = pd.DataFrame({'numeric': [1, 2, 3, 4, 5]})
    processed_standard = table_preprocessing(df, numeric_scaling='standard')

    scaler = StandardScaler()
    df['numeric'] = scaler.fit_transform(df[['numeric']]).flatten()

    expected_standard = df['numeric']
    np.testing.assert_allclose(processed_standard.flatten(), expected_standard.values, rtol=1e-6)

    processed_minmax = table_preprocessing(df, numeric_scaling='minmax')
    expected_minmax = (df['numeric'] - df['numeric'].min()) / (df['numeric'].max() - df['numeric'].min())
    np.testing.assert_allclose(processed_minmax.flatten(), expected_minmax.values, rtol=1e-6)


def test_encoding_methods():
    """Тест методов кодирования категориальных признаков."""
    df = pd.DataFrame({'cat': ['a', 'b', 'a']})
    processed_onehot = table_preprocessing(df, categorical_encoding='one-hot')
    assert processed_onehot.shape == (3, 2)

    processed_label = table_preprocessing(df, categorical_encoding='label')
    expected_label = np.array([0, 1, 0])
    np.testing.assert_array_equal(processed_label.flatten(), expected_label)


def test_verbose_output(sample_data, capsys):
    """Тест вывода verbose-информации."""
    table_preprocessing(sample_data, verbose=True)
    captured = capsys.readouterr()
    assert "Предобработка" in captured.out
