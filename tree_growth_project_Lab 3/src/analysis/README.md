# Analysis

## features.py
Object-oriented feature extraction. Each sprout is treated as a discrete signal (its internode lengths from base to tip).

| Class / function | What it does |
|---|---|
| `FeatureExtractor` | Parent class: stores the sequence; `extract()` must be written by each child class. |
| `SpatialFeatures` | `max_length`, `mean_length`, `std_length`, `cv_length` (std / mean, normalized variability), `energy`, `rms`. |
| `FrequencyFeatures` | `low_freq_ratio`: share of the lowest frequency in the FFT spectrum (shape of the sprout). NaN for sprouts with less than 4 internodes or with all internodes equal. |
| `extract_features_loop` | Extracts the features row by row with `apply(axis=1)`. Unoptimized version. |
| `extract_features` | Extracts the features from one NumPy matrix (vectorized spatial features). Optimized version. |
| `add_features` | Adds the features as new columns of the dataset. |

## statistics.py
| Function | What it does |
|---|---|
| `group_test` | Shapiro-Wilk per treatment, then Kruskal-Wallis (or ANOVA if all groups are normal). |
| `compare_two_groups` | Kruskal-Wallis between two treatments (e.g. control vs. methanol_control). |
| `fdr_loop` | Fisher Discriminant Ratio filtering each treatment inside a loop. Unoptimized version. |
| `fdr` | Fisher Discriminant Ratio with one `groupby` pass. Optimized version. |
| `auc_one_vs_rest` | ROC-AUC of each feature for each treatment against the rest. |
| `auc_control_vs_treated` | ROC-AUC of each feature for control vs. treated sprouts. |
| `threshold_rule` | Accuracy, precision, recall and F1 of the rule "feature > threshold → control". |

## How to use it
```python
from src.analysis.features import add_features, FEATURE_COLS
from src.analysis import statistics as st
df = add_features(df)
st.fdr(df, FEATURE_COLS)
```

## Complexity
`complexity_analysis.ipynb` contains the line-by-line and global complexity of every function and the comparison of the two optimizations (feature extraction and FDR).
