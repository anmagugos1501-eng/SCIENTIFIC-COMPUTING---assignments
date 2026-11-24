"""
Feature extraction module (object-oriented).

Each sprout is treated as a discrete signal: the sequence of its internode
lengths from base to tip. Features are extracted in two domains:

- Spatial domain: size and variability of the internodes.
- Frequency domain: shape of the sprout, from its FFT.
"""

import numpy as np
import pandas as pd
import scipy.fft

INTERNODE_COLS = [f'INTERNODE_{i}' for i in range(1, 30)]
FEATURE_COLS = ['max_length', 'mean_length', 'std_length', 'cv_length',
                'energy', 'rms', 'low_freq_ratio']


# Parent class: it only saves the internode sequence of one sprout.
# extract() is empty here. Each child class writes its own version
class FeatureExtractor:
    """
    Parent class for the feature extractors.

    It only stores the internode sequence of one sprout. Each child class
    writes its own version of extract().

    Parameters
    ----------
    seq : numpy.ndarray
        Internode lengths of one sprout, from base to tip, without NaN.
    """

    def __init__(self, seq):
        self.seq = seq

    def extract(self):
        """
        Compute the features of the sequence.

        Raises
        ------
        NotImplementedError
            Always, because the parent class has no rule.
        """
        raise NotImplementedError('no rule in class %s' %
                                  self.__class__.__name__)


# Child class 1 is the spatial domain, it has the features about the size of the internodes
class SpatialFeatures(FeatureExtractor):
    """
    Spatial-domain features: size and variability of the internodes.
    """

    def extract(self):
        """
        Compute the spatial features of the sequence.

        cv_length is the normalized version of std_length (std / mean), so it
        measures how uneven the internodes are without depending on the size
        of the sprout.

        Returns
        -------
        dict
            max_length, mean_length, std_length, cv_length, energy and rms.
        """
        mean = np.mean(self.seq)
        std = np.std(self.seq)
        return {'max_length': np.max(self.seq),
                'mean_length': mean,
                'std_length': std,
                # cv is std divided by mean, I added it to normalize std like the professor asked in Lab 2
                'cv_length': std / mean,
                'energy': np.sum(self.seq**2),
                'rms': np.sqrt(np.mean(self.seq**2))}


# Child class 2 is the frequency domain, it has the feature about the shape of the sprout
class FrequencyFeatures(FeatureExtractor):
    """
    Frequency-domain feature: shape of the sprout from its FFT.
    """

    def extract(self):
        """
        Compute the low-frequency ratio of the sequence.

        It is the share of the lowest frequency in the energy of the first half
        of the spectrum (without the DC value): close to 1 means a smooth arch,
        close to 0 means a zigzag. Sprouts with less than 4 internodes, or with
        all internodes equal, have no shape to analyze and get NaN.

        Returns
        -------
        dict
            low_freq_ratio.
        """
        if len(self.seq) < 4 or np.std(self.seq) < 1e-10:
            return {'low_freq_ratio': np.nan}
        spectrum = np.abs(scipy.fft.fft(self.seq))
        half = len(self.seq) // 2
        low_freq_ratio = spectrum[1]**2 / np.sum(spectrum[1:half+1]**2)
        return {'low_freq_ratio': low_freq_ratio}


def extract_one(seq):
    """
    Compute all the features of one sprout using both extractor classes.

    Parameters
    ----------
    seq : numpy.ndarray
        Internode lengths of one sprout, without NaN.

    Returns
    -------
    dict
        The seven features of the sprout.
    """
    features = {}
    # Use both classes and join their results in one dictionary, this is polymorphism
    for Extractor in SpatialFeatures, FrequencyFeatures:
        features.update(Extractor(seq).extract())
    return features


def extract_features_loop(df):
    """
    Extract the features of every sprout row by row (unoptimized version).

    It uses DataFrame.apply with axis=1, as in the Lab 2 notebook: for each
    row pandas builds a Series, removes the NaN and converts it to an array.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean dataset.

    Returns
    -------
    pandas.DataFrame
        One row per sprout and one column per feature.
    """
    def row_features(row):
        """
        Compute the features of one row of the dataset, that is one sprout.
        """
        # dropna removes the internodes that don't exist, to_numpy() turns it into an array
        seq = row[INTERNODE_COLS].dropna().to_numpy().astype(float)
        return pd.Series(extract_one(seq))

    return df.apply(row_features, axis=1)[FEATURE_COLS]


def extract_features(df):
    """
    Extract the features of every sprout (optimized version).

    The internode columns are converted to one NumPy matrix only once. The
    spatial features are computed for all sprouts at the same time with
    vectorized NaN-aware NumPy functions, and the frequency feature is computed
    over NumPy rows, without building a pandas Series for each sprout.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean dataset.

    Returns
    -------
    pandas.DataFrame
        One row per sprout and one column per feature (same values as
        extract_features_loop).
    """
    X = df[INTERNODE_COLS].to_numpy(dtype=float)
    # np.nanmean, np.nanstd, np.nanmax and np.nansum work like np.mean and np.std, but they skip the NaN, so I can use them on the whole matrix at once
    mean = np.nanmean(X, axis=1)
    std = np.nanstd(X, axis=1)
    features = pd.DataFrame({'max_length': np.nanmax(X, axis=1),
                             'mean_length': mean,
                             'std_length': std,
                             'cv_length': std / mean,
                             'energy': np.nansum(X**2, axis=1),
                             'rms': np.sqrt(np.nanmean(X**2, axis=1))},
                            index=df.index)
    features['low_freq_ratio'] = [
        FrequencyFeatures(row[~np.isnan(row)]).extract()['low_freq_ratio']
        for row in X]
    return features[FEATURE_COLS]


def add_features(df, optimized=True):
    """
    Add the extracted features as new columns of the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean dataset.
    optimized : bool, optional
        If True, use extract_features; if False, extract_features_loop.

    Returns
    -------
    pandas.DataFrame
        Dataset with the seven feature columns added.
    """
    features = extract_features(df) if optimized else extract_features_loop(df)
    return pd.concat([df, features], axis=1)
