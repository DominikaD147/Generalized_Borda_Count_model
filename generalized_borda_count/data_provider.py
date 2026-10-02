from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from ucimlrepo import fetch_ucirepo


class DatasetName(Enum):
    IRIS = "iris"
    WINE = "wine"
    ZOO = "zoo"
    GLASS = "glass.csv"
    VEHICLE = "vehicle.csv"

DATA_PATH = Path(__file__).parent.parent / "data"

def _load_from_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param filename: .csv file path
    :return: X, y - numpy arrays
    """
    path = DATA_PATH / filename

    if not path.exists():
        raise FileNotFoundError(f"{path} file does not exist in directory {DATA_PATH}")

    df = pd.read_csv(path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1]

    # convert text labels to numbers
    if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]
    else:
        y = y.to_numpy()

    return X, y


def get_dataset(name: DatasetName) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset must have numeric values but can have categorical labels.
    Categorical labels are converted to numeric values.
    :param name: dataset name - enum
    :return: X, y - numpy arrays
    """
    if name == DatasetName.IRIS:
        return load_iris(return_X_y=True)
    elif name == DatasetName.WINE:
        return load_wine(return_X_y=True)
    elif name == DatasetName.ZOO:
        zoo = fetch_ucirepo(id=111)
        X_df = zoo.data.features
        y_df = zoo.data.targets

        if 'animal_name' in X_df.columns:
            X_df = X_df.drop(columns=['animal_name'])

        X = X_df.values
        y = y_df.values.flatten()

        return X, y

    else:
        return _load_from_file(name.value)