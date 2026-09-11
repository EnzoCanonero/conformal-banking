# BANKING77 data

## Source

Use the official [PolyAI BANKING77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data),
pinned to revision `57ec275d8078af65b7731c2a98be812d844a6d6b`.
It contains 10,003 training examples, 3,080 test examples and 77 intents.

The dataset is distributed under [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE),
separately from this repository's MIT-licensed code. Dataset reference:
Casanueva et al. (2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807).

## Download

Run from the repository root:

```bash
mkdir -p data/raw/banking77

banking77_source="https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"

for filename in train.csv test.csv categories.json; do
    curl --fail --location "$banking77_source/$filename" \
        --output "data/raw/banking77/$filename"
done
```

Downloaded files are ignored by Git. Loading data never triggers a download.
The CSV files contain `text` and `category` columns; `categories.json` defines
the shared label order.

## Load and split

Install the existing scikit-learn extra with `python -m pip install -e ".[example]"`.
Then load the data in Python:

```python
from conformal_selective_prediction.data import load_banking77

data = load_banking77("data/raw/banking77", random_seed=42)

print(f"Training examples: {data.train.labels.size}")
print(f"Calibration examples: {data.calibration.labels.size}")
print(f"Test examples: {data.test.labels.size}")
```

- Split only the official training data into 7,502 training and 2,501
  calibration examples, stratified by intent.
- Preserve the official test examples and their order. Never use them for
  fitting preprocessing, training or choosing thresholds.
- Use the same seed to reproduce the split; change it for later repeated-split
  experiments. The official test partition stays fixed.
- Fit preprocessing and the classifier only on `data.train`. Reserve
  `data.calibration` for conformal calibration.

Each partition contains aligned `texts`, integer `labels` and `sample_ids`.
IDs such as `train:0` and `test:0` identify zero-based records in the original
CSV files and are retained through splitting.

Label `i` always refers to `data.class_names[i]`, following `categories.json`,
not alphabetical order or the order of rows in a split. Later classifiers must
align their probability columns with these indices before conformal scoring.

## Split limitations

Partition disjointness refers to original records, not semantic uniqueness.
The official files have no exact text overlap, but six text values occur in
both official splits after lowercasing and trimming whitespace. Preserve the
official data rather than silently deduplicating it, and report this upstream
overlap when interpreting the baseline results.
