import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
)


def load_dataframe(input_path: str) -> pd.DataFrame:
    """Load a DataFrame from a file."""
    if input_path.endswith(".csv"):
        peek_df = pd.read_csv(input_path, nrows=1)
        if peek_df.columns[0].startswith("Unnamed") or peek_df.columns[0].isdigit():
            df = pd.read_csv(input_path, index_col=0)
        else:
            df = pd.read_csv(input_path)
    elif input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path, index_col=None)
    elif input_path.endswith(".pickle"):
        df = pd.read_pickle(input_path)
    elif input_path.endswith(".json"):
        df = pd.read_json(input_path)
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".hdf") or input_path.endswith(".h5"):
        df = pd.read_hdf(input_path, index_col=None)
    else:
        supported_formats = ", ".join(
            [
                "CSV",
                "Excel (.xlsx)",
                "Pickle",
                "JSON",
                "Parquet",
                "HDF5 (.hdf, .h5)",
            ]
        )
        raise ValueError(
            f"The file format is not supported. Please convert your file to one of the following supported formats: {supported_formats}."
        )
    return df


def table_preprocessing(
    input_dataframe: pd.DataFrame,
    numeric_columns: List[str] = [],
    categorical_columns: List[str] = [],
    target_columns: List[str] = [],
    ignore_columns: List[str] = [],
    unknown_column_action: str = "infer",
    numeric_threshold: float = 0.05,
    numeric_scaling: str = "standard",
    categorical_encoding: str = "one-hot",
    nan_action: str = "infer",
    nan_threshold: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Кодирование табличных данных"""

    if verbose:
        print("Предобработка")

    # Загрузка датафрейма
    df = load_dataframe(input_dataframe, verbose=verbose)

    print(df.shape, df.dtypes)

    # Checking target_columns
    if target_columns is not None:
        if isinstance(target_columns, list):
            if target_columns in df.columns is False:
                raise ValueError(f"Target column {target_columns} not found.")
            target_columns = [target_columns]  # We need them to be lists.
        else:
            for target_col in target_columns:
                if target_col in df.columns is False:
                    raise ValueError(f"Target column {target_col} not found.")
    else:
        target_columns = []

    # Checking numeric_columns
    if numeric_columns is not None:
        if isinstance(target_columns, list):
            if numeric_columns in df.columns is False:
                raise ValueError(f"Numeric column {numeric_columns} not found.")
            numeric_columns = [numeric_columns]  # We need them to be lists.
        else:
            for numeric_col in numeric_columns:
                if numeric_col in df.columns is False:
                    raise ValueError(f"Numeric column {numeric_col} not found.")
    else:
        numeric_columns = []

    # Checking categorical_columns
    if categorical_columns is not None:
        if isinstance(target_columns, list):
            if categorical_columns in df.columns is False:
                raise ValueError(f"Categorical column {categorical_columns} not found.")
            categorical_columns = [categorical_columns]  # We need them to be lists.
        else:
            for categorical_col in categorical_columns:
                if categorical_col in df.columns is False:
                    raise ValueError(f"Categorical column {categorical_col} not found.")
    else:
        categorical_columns = []

    # Checking ignore_columns
    if ignore_columns is not None:
        if isinstance(target_columns, list):
            if ignore_columns in df.columns is False:
                raise ValueError(f"Ignore column {ignore_columns} not found.")
            ignore_columns = [ignore_columns]  # We need them to be lists.
        else:
            for ignore_col in ignore_columns:
                if ignore_col in df.columns is False:
                    raise ValueError(f"Ignore column {ignore_col} not found.")
    else:
        ignore_columns = []

    # Targets should not be preprocessed
    ignore_columns += target_columns

    # Unknown columns inference
    if unknown_column_action == "infer":
        for col in df.columns:
            if (
                col not in numeric_columns
                and col not in categorical_columns
                and col not in ignore_columns
            ):
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    numeric_columns.append(col)
                    if verbose:
                        print(
                            f"{datetime.datetime.now()}: Column '{col}' added to numeric columns by inference."
                        )
                elif df[col].dtype == "bool" or np.issubdtype(
                    df[col].dtype, np.datetime64
                ):
                    ignore_columns.append(col)
                    if verbose:
                        print(
                            f"{datetime.datetime.now()}: Column '{col}' added to ignored columns by inference."
                        )
                elif df[col].dtype == "object":
                    categorical_columns.append(col)
                    if verbose:
                        print(
                            f"{datetime.datetime.now()}: Column '{col}' added to categorical column columns by inference."
                        )
                else:
                    unique_ratio = len(df[col].unique()) / len(df[col])
                    if unique_ratio > numeric_threshold:
                        numeric_columns.append(col)
                        if verbose:
                            print(
                                f"{datetime.datetime.now()}: Column '{col}' added to numeric columns by unique ratio inference."
                            )
                    else:
                        categorical_columns.append(col)
                        if verbose:
                            print(
                                f"{datetime.datetime.now()}: Column '{col}' added to categorical columns by unique ratio inference."
                            )
    elif unknown_column_action == "ignore":
        ignore_columns += [
            col
            for col in df.columns
            if col not in numeric_columns
            and col not in categorical_columns
            and col not in ignore_columns
        ]
    else:
        raise ValueError(
            f"unknown_column_action {unknown_column_action} not supported. Aborting..."
        )
    if verbose:
        print("Dataframe short report\n")
        print(f"{df.shape[0]} rows and {df.shape[1]} columns")
        print(f"column list: {list(df.columns)}")
        print(f"nans:\n{df.isna().sum()}")

    # Set target columns to be only one colum
    target_col_name = (
        tuple(target_columns)
        if len(target_columns) > 1
        else (target_columns[0] if len(target_columns) == 1 else "")
    )
    if len(target_columns) > 1:
        df[target_col_name] = df[target_columns].apply(tuple, axis=1)
        df = df.drop(columns=target_columns)

    if len(target_columns) != 0:
        unique_targets = np.unique(df[target_col_name].values)
        N_col = df.shape[0]
        print("Target class proportions")
        for target in unique_targets:
            n_target = df[df[target_col_name] == target].shape[0]
            print(f"\t{target}: {n_target / N_col * 100}%")
    print("--------------------------\nEnd of the report.")

    # NaNs
    if nan_action == "drop row":
        df.dropna(inplace=True)
        if verbose:
            print(f"{datetime.datetime.now()}: Dropped rows with NaN values.")
    elif nan_action == "drop column":
        df.dropna(axis=1, thresh=int(nan_threshold * df.shape[0]), inplace=True)
        if verbose:
            print(
                f"{datetime.datetime.now()}: Dropped columns with NaN values above threshold."
            )
    elif nan_action == "infer":
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mean())
            if verbose:
                print(
                    f"{datetime.datetime.now()}: Filled NaN values in numeric column '{col}' with mean."
                )
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            if verbose:
                print(
                    f"{datetime.datetime.now()}: Filled NaN values in categorical column '{col}' with mode."
                )
        if verbose:
            print(f"{datetime.datetime.now()}: Filled NaN values with column means.")

    # Preprocessing numerical cols
    if numeric_scaling == "standard":
        scaler = StandardScaler()
    elif numeric_scaling == "minmax":
        scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    if verbose:
        print(
            f"{datetime.datetime.now()}: Scaled numeric columns using {numeric_scaling} scaling."
        )

    # Preprocessing cat cols
    if categorical_encoding == "one-hot":
        df = pd.get_dummies(df, columns=categorical_columns)
    elif categorical_encoding == "label":
        encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = encoder.fit_transform(df[col])
    if verbose:
        print(
            f"{datetime.datetime.now()}: Encoded categorical columns using {categorical_encoding} encoding."
        )

    print(df.shape, df.dtypes)

    return df.to_numpy()
