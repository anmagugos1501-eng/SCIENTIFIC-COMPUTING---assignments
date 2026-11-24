"""
Run the whole Tree Growth pipeline.

Usage (from the project root):
    python main.py

It reads data/raw/cjs.csv, saves the clean data and the features in
data/processed/, the result tables in results/tables/ and the figures in
results/figures/.
"""

import os
import matplotlib
matplotlib.use('Agg')

from src.preprocessing.cleaning import preprocess
from src.analysis.features import add_features, FEATURE_COLS
from src.analysis import statistics as st
from src.visualization import plots

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'data', 'raw', 'cjs.csv')
PROCESSED = os.path.join(ROOT, 'data', 'processed')
TABLES = os.path.join(ROOT, 'results', 'tables')
FIGURES = os.path.join(ROOT, 'results', 'figures')


def main():
    """
    Run preprocessing, analysis and visualization, and save all the outputs.
    """
    # os is used to build the folder paths and to create the folders if they don't exist
    for folder in (PROCESSED, TABLES, FIGURES):
        os.makedirs(folder, exist_ok=True)

    # 1. Preprocessing
    df = preprocess(RAW)
    df.to_csv(os.path.join(PROCESSED, 'cjs_clean.csv'), index=False)

    # 2. Feature extraction
    df = add_features(df)
    df.to_csv(os.path.join(PROCESSED, 'cjs_features.csv'), index=False)

    # 3. Statistical analysis and separability
    st.group_test(df, FEATURE_COLS).to_csv(
        os.path.join(TABLES, 'kruskal_wallis.csv'))
    st.compare_two_groups(df, FEATURE_COLS, 4, 5).to_csv(
        os.path.join(TABLES, 'control_vs_methanol.csv'))
    st.fdr(df, FEATURE_COLS).to_csv(
        os.path.join(TABLES, 'fdr.csv'), header=['FDR'])
    st.auc_one_vs_rest(df, FEATURE_COLS).to_csv(
        os.path.join(TABLES, 'auc_one_vs_rest.csv'))
    st.auc_control_vs_treated(df, FEATURE_COLS).to_csv(
        os.path.join(TABLES, 'auc_control_vs_treated.csv'), header=['AUC'])
    st.threshold_rule(df, 'std_length', [0.3, 0.5, 0.75, 1.0]).to_csv(
        os.path.join(TABLES, 'threshold_rule.csv'), index=False)

    # 4. Visualization
    plots.boxplot_tl_in(df, os.path.join(FIGURES, 'boxplot_tl_in.png'))
    plots.boxplot_features(df, FEATURE_COLS,
                           os.path.join(FIGURES, 'boxplot_features.png'))
    plots.boxplot_control_vs_treated(
        df, FEATURE_COLS, os.path.join(FIGURES, 'boxplot_control_vs_treated.png'))
    plots.median_barplot(df, FEATURE_COLS,
                         os.path.join(FIGURES, 'median_barplot.png'))
    plots.radar_plot(df, FEATURE_COLS, os.path.join(FIGURES, 'radar_plot.png'))

    print('Pipeline finished. Outputs saved in data/processed/ and results/.')


# this makes main run only when I run this file, not when I import it
if __name__ == '__main__':
    main()
