"""
Preprocessing module for the Tree Growth dataset.

This module loads the raw data, cleans it and prepares it for the analysis:
missing values marked with '?', type conversion, categorical encoding and
the consistency check between IN and the internode columns.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


INTERNODE_COLS = [f'INTERNODE_{i}' for i in range(1, 30)]
CATEGORICAL_COLS = ['TR', 'TREE', 'BR']


def load_data(path):
    """
    Load the raw Tree Growth dataset.

    Parameters
    ----------
    path : str
        Path or URL of the raw CSV file.

    Returns
    -------
    pandas.DataFrame
        Raw dataset, exactly as it is stored in the file.
    """
    return pd.read_csv(path)


def replace_missing_loop(df):
    """
    Replace the '?' marks with NaN, cell by cell (unoptimized version).

    It goes through every column and every cell with apply + map, as in the
    Lab 2 notebook.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset with NaN instead of '?'.
    """
    # Replace the '?' marks with NaN so pandas recognizes them as missing values
    return df.apply(lambda col: col.map(lambda x: np.nan if x == '?' else x))


def replace_missing(df):
    """
    Replace the '?' marks with NaN in one vectorized step (optimized version).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset with NaN instead of '?'.
    """
    # I used replace here to get the vectorized version of apply + map, I took this idea from the notebook S7
    return df.replace('?', np.nan)


def convert_internodes(df):
    """
    Convert the internode columns from text to float.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset where the '?' marks were already replaced with NaN.

    Returns
    -------
    pandas.DataFrame
        Dataset with the 29 internode columns as float.
    """
    df = df.copy()
    # Convert the internode columns from text to numbers
    df[INTERNODE_COLS] = df[INTERNODE_COLS].astype(float)
    return df


def encode_categories(df):
    """
    Convert the label columns to category and encode the treatment.

    TR, TREE and BR are labels, not measurements, so they are converted to the
    category type. TR is the class variable, so it is also encoded as integers
    from 0 to 5 (alphabetical order) in the new column TR_code.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the internode columns already converted.

    Returns
    -------
    pandas.DataFrame
        Dataset with categorical columns and the new column TR_code.
    """
    df = df.copy()
    # Convert the label columns, that are treatment, tree and branch, to the category type
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype('category')
    # Encode the treatment, class variable, as integers from 0 to 5
    label_encoder = LabelEncoder()
    df['TR_code'] = label_encoder.fit_transform(df['TR'])
    return df


def find_inconsistent(df):
    """
    Find the sprouts where IN does not match the number of internodes with data.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the internode columns as float.

    Returns
    -------
    pandas.DataFrame
        Rows where IN is different from the number of non-missing internodes.
    """
    # Check if the data makes sense: IN says how many internodes each sprout has,
    # so I count the internodes that have a value and compare both numbers
    in_count = df[INTERNODE_COLS].notna().sum(axis=1)
    return df[df['IN'] != in_count]


def remove_inconsistent(df):
    """
    Remove the sprouts where IN does not match the internode data.

    In this dataset only one sprout (N = 1067) is inconsistent. TL cannot tell
    which value is correct, so the sprout is removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the internode columns as float.

    Returns
    -------
    pandas.DataFrame
        Dataset without the inconsistent sprouts.
    """
    # Only one sprout, N = 1067, has a problem: IN says 3, but it has 4 internodes.
    # TL can't tell me which one is right, so I remove this sprout, it's only 1 of 2796
    bad = find_inconsistent(df)
    return df.drop(index=bad.index)


def preprocess(path, optimized=True):
    """
    Run the whole preprocessing pipeline.

    Parameters
    ----------
    path : str
        Path or URL of the raw CSV file.
    optimized : bool, optional
        If True, use the vectorized replacement of '?'. Default True.

    Returns
    -------
    pandas.DataFrame
        Clean dataset, ready for feature extraction.
    """
    df = load_data(path)
    df = replace_missing(df) if optimized else replace_missing_loop(df)
    df = convert_internodes(df)
    df = encode_categories(df)
    df = remove_inconsistent(df)
    return df
