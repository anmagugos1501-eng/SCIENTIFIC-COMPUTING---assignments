"""
Visualization module.

Functions to generate the figures of the project. The violin and scatter plots
of Lab 2 were replaced because they do not work well with this dataset: most
treated sprouts are concentrated near zero with a long tail. Boxplots on a
log(1 + x) scale, median bar plots and a radar plot are used instead.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Clear titles and axis labels for each feature, used in all the plots
TITLES = {
    'max_length': 'Longest internode of each sprout',
    'mean_length': 'Average internode length of each sprout',
    'std_length': 'Variability of internode lengths (std)',
    'cv_length': 'Normalized variability (std / mean)',
    'energy': 'Energy of the internode sequence',
    'rms': 'RMS of internode lengths',
    'low_freq_ratio': 'Sprout shape: share of smooth (arch-like) variation'
}

UNITS = {
    'max_length': 'Length (cm)',
    'mean_length': 'Length (cm)',
    'std_length': 'Standard deviation (cm)',
    'cv_length': 'Coefficient of variation',
    'energy': 'Energy (cm²)',
    'rms': 'RMS (cm)',
    'low_freq_ratio': 'Low-frequency ratio (0 to 1)'
}

# Features with a long right tail, they are shown with the log of 1 + x
LOG_FEATURES = ['max_length', 'mean_length', 'std_length', 'energy', 'rms']


def _save(fig, path):
    """
    Save a figure if a path is given.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or None
        Output path. If None, the figure is not saved.
    """
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches='tight')


def boxplot_tl_in(df, path=None):
    """
    Boxplots of total length (TL) and number of internodes (IN) by treatment.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean dataset.
    path : str, optional
        Where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Boxplots to compare TL and IN between treatments and look for outliers
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(axes, ['TL', 'IN'],
                              ['Total length (TL) by treatment',
                               'Number of internodes (IN) by treatment']):
        sns.boxplot(x='TR', y=col, data=df, hue='TR', palette='Set2',
                    legend=False, dodge=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Treatment')
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    _save(fig, path)
    return fig


def boxplot_features(df, features, path=None):
    """
    Boxplots of each feature by treatment.

    Skewed features are drawn as log(1 + x), so the treated groups, which are
    concentrated near zero, can be compared with the controls.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the features.
    features : list of str
        Features to plot.
    path : str, optional
        Where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_rows = int(np.ceil(len(features) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        data = df[['TR', feature]].dropna().copy()
        label = UNITS[feature]
        if feature in LOG_FEATURES:
            # log1p means log of 1 + x, I used it because most values are near zero and with a normal scale the boxes looked flat
            data[feature] = np.log1p(data[feature])
            label = f'log(1 + {label})'
        sns.boxplot(x='TR', y=feature, data=data, hue='TR', palette='Set2',
                    legend=False, dodge=False, ax=ax)
        ax.set_title(TITLES[feature], fontsize=10)
        ax.set_xlabel('Treatment')
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=45)
    for ax in axes[len(features):]:
        ax.axis('off')
    fig.suptitle('Distribution of the extracted features by treatment',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, path)
    return fig


def boxplot_control_vs_treated(df, features, path=None):
    """
    Boxplots of each feature for control vs. treated sprouts.

    It replaces the scatter plot of Lab 2: it shows how much the two groups
    overlap for each feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the TR_code column and the features.
    features : list of str
        Features to plot.
    path : str, optional
        Where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = df.copy()
    data['group'] = np.where(data['TR_code'] >= 4, 'Control', 'Treated')
    n_rows = int(np.ceil(len(features) / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        values = data[['group', feature]].dropna().copy()
        label = UNITS[feature]
        if feature in LOG_FEATURES:
            # log1p means log of 1 + x, I used it because most values are near zero and with a normal scale the boxes looked flat
            values[feature] = np.log1p(values[feature])
            label = f'log(1 + {label})'
        sns.boxplot(x='group', y=feature, data=values, hue='group',
                    palette='Set1', legend=False, dodge=False, ax=ax)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel(label)
    for ax in axes[len(features):]:
        ax.axis('off')
    fig.suptitle('Control vs. treated sprouts', fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, path)
    return fig


def median_barplot(df, features, path=None):
    """
    Bar plot of the median of each feature by treatment.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the features.
    features : list of str
        Features to plot.
    path : str, optional
        Where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Table with the median of each feature for each treatment
    medians = df.groupby('TR', observed=True)[features].median()
    # Bar plot of the medians, one subplot for each feature
    n_rows = int(np.ceil(len(features) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        bars = ax.bar(medians.index.astype(str), medians[feature],
                      color=sns.color_palette('Set2', len(medians)))
        ax.bar_label(bars, fmt='%.2f', fontsize=8)
        ax.set_title(TITLES[feature], fontsize=10)
        ax.set_ylabel(UNITS[feature])
        ax.tick_params(axis='x', rotation=45)
    for ax in axes[len(features):]:
        ax.axis('off')
    fig.suptitle('Median of each extracted feature by treatment',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, path)
    return fig


def radar_plot(df, features, path=None):
    """
    Radar plot of the median feature profile of each treatment (as in S4).

    Each feature is divided by its maximum median, so all axes go from 0 to 1.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the features.
    features : list of str
        Features to plot.
    path : str, optional
        Where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Radar plot of the median profile of each treatment, each feature is divided by its maximum so it goes from 0 to 1
    medians = df.groupby('TR', observed=True)[features].median()
    radar = medians / medians.max()
    N = len(features)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], features, size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ['0.25', '0.50', '0.75'], color='grey',
               size=7)
    plt.ylim(0, 1)
    colors = sns.color_palette('Set2', len(radar))
    for i, treatment in enumerate(radar.index):
        values = radar.loc[treatment].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=str(treatment),
                color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    plt.title('Median feature profile by treatment (normalized by the maximum)',
              size=11, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    _save(fig, path)
    return fig
