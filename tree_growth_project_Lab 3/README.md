# Tree Growth: effect of growth regulators on maple sprouts

Final project of the course Scientific Computing (Master in Automation and Industrial Control, ITM).

This project formalizes the analysis of Lab 1 and Lab 2 as a structured pipeline: preprocessing, feature extraction and statistical analysis, and visualization. It also analyzes the computational complexity of the implemented algorithms and compares unoptimized and optimized versions.

## Project structure

```
tree_growth_project/
├── data/
│   ├── raw/                  # Original dataset (cjs.csv)
│   ├── processed/            # Clean data and extracted features (generated)
│   └── README.md
├── src/
│   ├── preprocessing/        # Loading and cleaning (cleaning.py) + complexity notebook
│   ├── analysis/             # Features (features.py), statistics (statistics.py) + complexity notebook
│   └── visualization/        # Figures (plots.py) + complexity notebook
├── notebooks/
│   └── results_report.ipynb  # Explanation of the results
├── results/
│   ├── figures/              # Generated figures
│   ├── tables/               # Generated tables (CSV)
│   └── README.md
├── main.py                   # Runs the whole pipeline
├── requirements.txt
└── README.md
```

Each folder of `src/` contains its own complexity notebook, as required by the assignment, instead of keeping all notebooks in `notebooks/`. The folders `tests/`, `docs/` and the files `environment.yml` and `setup.py` of the reference layout were not included, because the project uses pip and is run as a script, not installed as a package.

## Setup

1. Install Python 3.10 or newer.
2. From the project root, install the libraries:
   ```
   pip install -r requirements.txt
   ```

## How to run

From the project root:
```
python main.py
```
This reads `data/raw/cjs.csv` and saves the clean data in `data/processed/`, the tables in `results/tables/` and the figures in `results/figures/`.

Then open the notebooks (for example, in VS Code or with `jupyter notebook`):
- `notebooks/results_report.ipynb`: explanation of the results.
- `src/<module>/complexity_<module>.ipynb`: complexity analysis of each module.

## Theoretical background

**The study.** The dataset comes from a case study published in the September 1995 issue of the *Canadian Journal of Statistics*, about the effect of two growth regulators, paclobutrazol (PP 333) and flurprimidol (EL-500), on the terminal sprouts that grow on trimmed silver maple trees [1]. Both compounds are known to inhibit the biosynthesis of gibberellins, the plant hormones that control the elongation of stems, so they are expected to produce shorter internodes [2].

**Features.** Each sprout is treated as a discrete signal: the sequence of its internode lengths from base to tip. In the spatial domain, the size and variability of this sequence are described with the maximum, mean, standard deviation, coefficient of variation (standard deviation divided by the mean, which removes the effect of size), energy and root mean square. In the frequency domain, the Fast Fourier Transform (FFT) [3] decomposes the sequence into frequencies; a sprout with a smooth arch shape concentrates its variation in the lowest frequency.

**Statistical tests and separability.** The Shapiro-Wilk test [4] checks whether a sample follows a normal distribution. When normality is not met, the Kruskal-Wallis test [5], a non-parametric alternative to one-way ANOVA based on ranks, compares several groups. The separability of the groups is measured with the Fisher Discriminant Ratio [6], which compares the dispersion between groups with the dispersion within them, and with the area under the ROC curve (AUC) [7], where 0.5 means random chance and 1.0 perfect separation.

**Computational complexity.** The O(n) notation describes how the execution time of an algorithm grows with the size of its input, ignoring constants and lower-order terms [8]. Two algorithms with the same order can still have very different execution times, because of the constant cost of each operation; vectorized code (NumPy, pandas) reduces this constant compared to Python loops.

## Methodology

1. **Preprocessing** (`src/preprocessing`): the `?` marks were replaced with NaN, the internode columns were converted to float, the label columns were converted to category, and the treatment was encoded as `TR_code` (0 to 5). The missing internodes were not filled, because they are internodes that do not exist in shorter sprouts. One inconsistent sprout (N = 1067, where IN did not match the internode data) was removed, leaving 2,795 sprouts.
2. **Feature extraction** (`src/analysis/features.py`): seven features were extracted with an object-oriented design (a parent class and one child class per domain). Following the feedback of Lab 2, the standard deviation was normalized as the coefficient of variation (`cv_length`).
3. **Statistical analysis and separability** (`src/analysis/statistics.py`): Shapiro-Wilk and Kruskal-Wallis tests among the six treatments, comparison of the two control groups, Fisher Discriminant Ratio, ROC-AUC (one-versus-rest and control vs. treated) and a simple threshold rule evaluated with accuracy, precision, recall and F1.
4. **Visualization** (`src/visualization`): boxplots on a log(1 + x) scale, median bar plots and a radar plot. The violin and scatter plots of Lab 2 were replaced because they did not work well with these skewed data.
5. **Complexity and optimization**: each module has a notebook with the line-by-line and global complexity of its functions. Three algorithms were optimized (replacement of missing values, feature extraction and FDR), and their execution times were compared for different input sizes with `timeit`.

## Summary of results

- The growth regulators strongly reduce the size of the sprouts: the median of the longest internode is 0.50 cm in the four treated groups, 1.30 cm in the control group and 3.95 cm in the methanol control.
- The internodes of the treated sprouts are also more uniform: the median of `cv_length` is 0 in all treated groups and 0.48 and 0.61 in the controls.
- The shape of the sprouts barely changes: the median of `low_freq_ratio` goes from 0.86 to 0.93 in all groups.
- Kruskal-Wallis found significant differences among treatments for all features, with very large statistics for the spatial features (436 to 737) and a small one for `low_freq_ratio` (16.5, p = 0.006).
- Fisher Discriminant Ratio: `cv_length` 0.338 (best), `std_length` 0.275, `max_length` 0.235, `rms` 0.230, `mean_length` 0.213, `energy` 0.065, `low_freq_ratio` 0.023.
- AUC control vs. treated: 0.82 with `std_length` and 0.81 with `cv_length`; one-versus-rest AUC for the treated groups between 0.50 and 0.64.
- Threshold rule with `std_length`: accuracy from 0.78 to 0.80 (baseline 0.754), best F1 of 0.573 with a threshold of 0.3.
- Optimizations: feature extraction about 50 times faster, FDR about 7 times faster and replacement of missing values about 4 times faster, with exactly the same results.

## Discussion

The results agree with the expected effect of the growth regulators: the treated sprouts are much shorter and their internodes are more uniform, while the overall arch shape of the sprouts is kept. The normalization of the standard deviation was useful: `std_length` was almost perfectly correlated with the size features, while `cv_length` measures the uniformity of the internodes independently of size, adds new information and has the highest Fisher Discriminant Ratio.

Although the differences between treatments are statistically significant, the separability is only moderate between control and treated sprouts and weak between the four treatments. The low values of the Fisher Discriminant Ratio show that the groups overlap a lot from sprout to sprout. The accuracy of the threshold rule is misleading because of the class imbalance (75 % of the sprouts are treated), so precision, recall and F1 are more informative.

The study has some limitations. Each treatment has only nine or ten trees, the sprouts from the same tree are not independent, and the variability between trees is high, which probably explains the unexpected difference between the two control groups. In addition, the frequency feature could only be computed for 1,021 sprouts, because most sprouts have less than four internodes.

Regarding complexity, the three optimizations keep the same order in n, O(n) for the feature extraction and the FDR when the number of features and treatments is fixed, but reduce the constant cost per sprout. In the FDR, the optimized version also removes the factor K (number of treatments), going from O(F · K · n) to O(F · n).

## Conclusions

- The growth regulators PP 333 and EL-500 reduce the size of the sprouts and make their internodes more uniform, but barely change their shape.
- The normalized variability of the internodes (`cv_length`) is the feature that best separates the treatments.
- Control and treated sprouts can be distinguished moderately; the specific treatments and doses are very similar to each other.
- Organizing the code into modules made the analysis reproducible with one command (`python main.py`), and vectorized operations reduced the execution time of the main algorithms by up to about 50 times without changing the results.

## IA declaration

I used artificial intelligence—specifically Claude, by Anthropic—as a translation tool and as support for organizing the project during the preprocessing phase. It helped me draft and refine code snippets that I wasn’t sure how to develop, guiding me on how to structure and fully implement them. Additionally, I used it to draft and translate README files, as well as to clarify concepts that I didn’t quite understand.

## References

[1] Case study on the effects of growth regulators on silver maple sprouts, *Canadian Journal of Statistics*, September 1995. Dataset provided by Dr. Fernando Camacho (Ontario Hydro Technologies). Available in OpenML as dataset `cjs` (ID 473): https://www.openml.org/d/473

[2] W. Rademacher, "Growth retardants: effects on gibberellin biosynthesis and other metabolic pathways," *Annual Review of Plant Physiology and Plant Molecular Biology*, vol. 51, pp. 501–531, 2000.

[3] J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series," *Mathematics of Computation*, vol. 19, no. 90, pp. 297–301, 1965.

[4] S. S. Shapiro and M. B. Wilk, "An analysis of variance test for normality (complete samples)," *Biometrika*, vol. 52, no. 3–4, pp. 591–611, 1965.

[5] W. H. Kruskal and W. A. Wallis, "Use of ranks in one-criterion variance analysis," *Journal of the American Statistical Association*, vol. 47, no. 260, pp. 583–621, 1952.

[6] R. A. Fisher, "The use of multiple measurements in taxonomic problems," *Annals of Eugenics*, vol. 7, no. 2, pp. 179–188, 1936.

[7] T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006.

[8] T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 3rd ed. MIT Press, 2009.
