# Visualization

`plots.py` generates the figures of the project. The violin and scatter plots of Lab 2 were replaced, because they do not work well with this dataset (most treated sprouts are concentrated near zero with a long tail). Skewed features are drawn on a log(1 + x) scale.

| Function | Figure |
|---|---|
| `boxplot_tl_in` | Boxplots of total length and number of internodes by treatment. |
| `boxplot_features` | Boxplots of each feature by treatment. |
| `boxplot_control_vs_treated` | Boxplots of each feature for control vs. treated sprouts (overlap between groups). |
| `median_barplot` | Median of each feature by treatment. |
| `radar_plot` | Median feature profile of each treatment, normalized by the maximum. |

Every function returns the figure and, if a `path` is given, saves it.

## Complexity
`complexity_visualization.ipynb` contains the line-by-line and global complexity of every function and its execution time.
