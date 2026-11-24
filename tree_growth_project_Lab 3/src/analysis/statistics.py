"""
Statistical analysis and separability module.

It contains the hypothesis tests (Shapiro-Wilk, Kruskal-Wallis / ANOVA), the
Fisher Discriminant Ratio (FDR), the ROC-AUC tables and the metrics of a
simple threshold classification rule.
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, shapiro, kruskal
from sklearn.metrics import roc_auc_score

TREATMENTS = {0: 'EL_500_20g/L', 1: 'EL_500_4g/L', 2: 'PP_333_20g/L',
              3: 'PP_333_4g/L', 4: 'control', 5: 'methanol_control'}


def group_test(df, features, alpha_normal=0.05):
    """
    Compare the six treatments for each feature.

    Same rule as in S5: Shapiro-Wilk is applied to each group, and if any group
    is not normal Kruskal-Wallis is used; otherwise, one-way ANOVA.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the TR_code column and the features.
    features : list of str
        Features to test.
    alpha_normal : float, optional
        Significance level of the normality test. Default 0.05.

    Returns
    -------
    pandas.DataFrame
        For each feature: test used, statistic and p-value.
    """
    # Shapiro-Wilk checks if each treatment is normal, then I use ANOVA or Kruskal-Wallis
    # Same rule as in S5: if any group is not normal, use Kruskal-Wallis
    rows = []
    for feature in features:
        # Split the sprouts into the 6 treatments
        groups = [df[df['TR_code'] == i][feature].dropna() for i in range(6)]
        shapiro_p = [shapiro(g).pvalue for g in groups]
        if min(shapiro_p) < alpha_normal:
            test = 'Kruskal-Wallis'
            stat, p_value = kruskal(*groups)
        else:
            test = 'ANOVA'
            stat, p_value = f_oneway(*groups)
        rows.append({'feature': feature, 'test': test,
                     'statistic': stat, 'p_value': p_value})
    return pd.DataFrame(rows).set_index('feature')


def compare_two_groups(df, features, code_a, code_b):
    """
    Compare two treatments with Kruskal-Wallis (non-normal data).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the TR_code column and the features.
    features : list of str
        Features to compare.
    code_a, code_b : int
        TR_code of the two treatments (e.g. 4 = control, 5 = methanol_control).

    Returns
    -------
    pandas.DataFrame
        For each feature: statistic and p-value.
    """
    # Compare the two control groups, control and methanol_control.
    # Methanol is only the solvent, so in theory both groups should be the same.
    # Kruskal-Wallis with 2 groups is used because the data are not normal
    a = df[df['TR_code'] == code_a]
    b = df[df['TR_code'] == code_b]
    rows = []
    for feature in features:
        stat, p_value = kruskal(a[feature].dropna(), b[feature].dropna())
        rows.append({'feature': feature, 'statistic': stat, 'p_value': p_value})
    return pd.DataFrame(rows).set_index('feature')


def fdr_loop(df, features):
    """
    Fisher Discriminant Ratio of each feature (unoptimized version).

    For each feature and each treatment, the rows of that treatment are
    filtered again from the whole dataset, as in the Lab 2 notebook.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the TR_code column and the features.
    features : list of str
        Features to evaluate.

    Returns
    -------
    pandas.Series
        FDR of each feature, sorted from highest to lowest.
    """
    # Fisher Discriminant Ratio for each feature, calculated one feature at a time.
    # FDR = dispersion between treatments / dispersion within treatments.
    # A high FDR means the groups are far apart and compact.
    # I calculate the FDR one feature at a time and not with matrices like in S5, because with matrices it gave negative values since the features are very correlated
    fdr = {}
    for feature in features:
        data = df[['TR_code', feature]].dropna()
        overall_mean = data[feature].mean()
        group_means = data.groupby('TR_code')[feature].mean()
        within = 0
        between = 0
        for k, mean_k in group_means.items():
            x = data[data['TR_code'] == k][feature]
            within += np.sum((x - mean_k)**2)
            between += len(x) * (mean_k - overall_mean)**2
        fdr[feature] = between / within
    return pd.Series(fdr).sort_values(ascending=False)


def fdr(df, features):
    """
    Fisher Discriminant Ratio of each feature (optimized version).

    One groupby pass computes the count, mean and variance of every feature in
    every treatment at the same time, so the data is not filtered again for
    each group. FDR = dispersion between treatments / dispersion within them.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the TR_code column and the features.
    features : list of str
        Features to evaluate.

    Returns
    -------
    pandas.Series
        FDR of each feature, sorted from highest to lowest (same values as
        fdr_loop).
    """
    # I get the count, mean and var of all the groups in one pass, so the data is not filtered again for each treatment
    grouped = df.groupby('TR_code')[features]
    count = grouped.count()
    mean = grouped.mean()
    # ddof=0 makes the variance use the same formula as fdr_loop
    var = grouped.var(ddof=0)
    overall_mean = (count * mean).sum() / count.sum()
    within = (count * var).sum()
    between = (count * (mean - overall_mean)**2).sum()
    return (between / within).sort_values(ascending=False)


def auc_one_vs_rest(df, features):
    """
    ROC-AUC of each feature for each treatment against the rest (as in S5).

    If the AUC is below 0.5, 1 - AUC is used, because the feature separates the
    groups in the opposite direction.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the TR_code column and the features.
    features : list of str
        Features to evaluate.

    Returns
    -------
    pandas.DataFrame
        One row per feature and one column per treatment.
    """
    # Empty table: one row per feature, one column per treatment
    table = pd.DataFrame(index=features, columns=list(TREATMENTS.values()),
                         dtype=float)
    for feature in features:
        data = df[['TR_code', feature]].dropna()
        # AUC one versus rest: each treatment is 1 and all the others are 0
        for class_idx, name in TREATMENTS.items():
            target = np.where(data['TR_code'] == class_idx, 1, 0)
            auc = roc_auc_score(target, data[feature])
            # If AUC < 0.5, use 1 - AUC, because the feature separates in the opposite direction
            table.loc[feature, name] = auc if auc >= .5 else 1 - auc
    return table


def auc_control_vs_treated(df, features):
    """
    ROC-AUC of each feature to separate control from treated sprouts.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the TR_code column and the features.
    features : list of str
        Features to evaluate.

    Returns
    -------
    pandas.Series
        AUC of each feature, sorted from highest to lowest.
    """
    # 1 is control, that is TR_code 4 and 5, and 0 is treated, that is TR_code 0 to 3
    is_control = np.where(df['TR_code'] >= 4, 1, 0)
    result = {}
    for feature in features:
        mask = df[feature].notna().to_numpy()
        auc = roc_auc_score(is_control[mask], df[feature][mask])
        result[feature] = auc if auc >= .5 else 1 - auc
    # Sort from best to worst
    return pd.Series(result).sort_values(ascending=False)


def threshold_rule(df, feature, thresholds):
    """
    Evaluate a simple rule: a sprout is a control if feature > threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the TR_code column and the feature.
    feature : str
        Feature used by the rule.
    thresholds : list of float
        Thresholds to test.

    Returns
    -------
    pandas.DataFrame
        TP, TN, FP, FN, accuracy, precision, recall and F1 for each threshold
        (control is the positive class).
    """
    y_true = np.where(df['TR_code'] >= 4, 1, 0)
    values = df[feature].to_numpy()
    results = []
    for threshold in thresholds:
        y_pred = np.where(values > threshold, 1, 0)
        TP = np.sum((y_pred == 1) & (y_true == 1))   # control predicted as control
        TN = np.sum((y_pred == 0) & (y_true == 0))   # treated predicted as treated
        FP = np.sum((y_pred == 1) & (y_true == 0))   # treated predicted as control
        FN = np.sum((y_pred == 0) & (y_true == 1))   # control predicted as treated
        accuracy = (TP + TN) / len(y_true)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        results.append({'threshold': threshold, 'TP': TP, 'TN': TN,
                        'FP': FP, 'FN': FN, 'accuracy': accuracy,
                        'precision': precision, 'recall': recall, 'F1': f1})
    return pd.DataFrame(results)
