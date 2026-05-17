"""
Real dataset pipelines for distributional regression benchmarks.

Each function returns a train/test split as numpy arrays.
Preprocessing is fitted on train only — no leakage.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_california_housing(
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Load and preprocess the California Housing dataset.

    Target: median house value (continuous, right-skewed).
    Features: 8 numerical features (location, demographics, housing stats).

    Parameters
    ----------
    test_size : float
        Fraction of data held out for testing.
    seed : int
        Random seed for train/test split.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, feature_names
    """
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Fit scaler on train only — never on test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(data.feature_names),
    }


def load_abalone(
    data_dir: str = "data/raw",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Load and preprocess the Abalone dataset (UCI).

    Target: number of rings (proxy for age); right-skewed count data.
    Features: physical measurements + sex (one-hot encoded).

    The raw CSV must be downloaded first — see download_abalone().

    Parameters
    ----------
    data_dir : str
        Directory containing abalone.csv.
    test_size : float
        Fraction of data held out for testing.
    seed : int
        Random seed for train/test split.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(
        f"{data_dir}/abalone.csv",
        header=None,
        names=[
            "sex",
            "length",
            "diameter",
            "height",
            "whole_weight",
            "shucked_weight",
            "viscera_weight",
            "shell_weight",
            "rings",
        ],
    )

    # One-hot encode sex (M, F, I)
    df = pd.get_dummies(df, columns=["sex"], drop_first=False)

    y = df["rings"].values.astype(float)
    X = df.drop(columns=["rings"]).values.astype(float)
    feature_names = [c for c in df.columns if c != "rings"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
    }


def download_abalone(data_dir: str = "data/raw") -> None:
    """
    Download the Abalone CSV from the UCI repository.

    Run this once before calling load_abalone().

    Parameters
    ----------
    data_dir : str
        Directory to save the downloaded file.
    """
    import urllib.request
    from pathlib import Path

    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/abalone/abalone.data"
    )
    path = Path(data_dir) / "abalone.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Abalone dataset to {path} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Done. {path.stat().st_size / 1024:.1f} KB saved.")
