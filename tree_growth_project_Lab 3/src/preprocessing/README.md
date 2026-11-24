# Preprocessing

`cleaning.py` loads the raw data and prepares it for the analysis.

| Function | What it does |
|---|---|
| `load_data` | Reads the raw CSV file. |
| `replace_missing_loop` | Replaces `?` with NaN cell by cell (`apply` + `map`). Unoptimized version. |
| `replace_missing` | Replaces `?` with NaN with `DataFrame.replace`. Optimized version. |
| `convert_internodes` | Converts the 29 internode columns from text to float. |
| `encode_categories` | Converts `TR`, `TREE` and `BR` to category and encodes `TR` as `TR_code` (0 to 5). |
| `find_inconsistent` | Finds the sprouts where `IN` does not match the number of internodes with data. |
| `remove_inconsistent` | Removes those sprouts (only N = 1067 in this dataset). |
| `preprocess` | Runs the whole preprocessing pipeline. |

## How to use it
From the project root:
```python
from src.preprocessing.cleaning import preprocess
df = preprocess('data/raw/cjs.csv')
```

## Complexity
`complexity_preprocessing.ipynb` contains the line-by-line and global complexity of every function and the comparison between `replace_missing_loop` and `replace_missing`.
