# TEMPEST — Ransomware Detection via API Call Sequences

This repository contains the experiment notebook used to produce the results reported
in the paper *"TEMPEST: Temporal Execution Modeling for Pre-Encryption Ransomware
Detection with Timestep-Level Explainability."*

The notebook trains a stacked Embedding-LSTM classifier on API call sequences,
compares it against Random Forest, SVM, CNN, and Feedforward baselines, runs several
ablation studies, and computes Integrated Gradients attributions for explainability.

## Contents

- `research-experiment.ipynb` — the full experiment pipeline (single master cell, run
  top to bottom)

## Requirements

Install the following Python packages before running:

```
tensorflow
scikit-learn
imbalanced-learn
scipy
pandas
numpy
matplotlib
seaborn
kaggle
```

You can install them with:

```bash
pip install tensorflow scikit-learn imbalanced-learn scipy pandas numpy matplotlib seaborn kaggle
```

If you hit compatibility errors, try installing without version pins first, and only
pin versions if something breaks — the code is written against fairly standard/stable
APIs (Keras `Sequential`, `sklearn` metrics, `imblearn.SMOTE`).

## Environment

This notebook was developed and run on **Kaggle Notebooks** with GPU acceleration
(2x Tesla T4). It uses hardcoded Kaggle paths (`/kaggle/working/...`) for reading and
writing files.

**Easiest way to run it: use Kaggle.**

If you want to run it elsewhere (Google Colab, local machine, other cloud notebook),
you will need to either:
- Create the same folder structure (`/kaggle/working/data/raw`,
  `/kaggle/working/data/raw2`, `/kaggle/working/data/splits`,
  `/kaggle/working/results`, `/kaggle/working/figures`), or
- Do a find-and-replace of `/kaggle/working/` with your own working directory path
  throughout the notebook.

A GPU is strongly recommended. The notebook trains dozens of models in total
(main model + 4 baselines + 5 ablation variants + 10 truncation-length models + 30
models for the 15-seed statistical test), so CPU-only execution will be very slow.

## Data

The notebook automatically downloads its two datasets from Kaggle using the Kaggle
API, so no manual download is needed — you just need Kaggle API credentials set up
(see below).

Datasets used:
- https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences
- https://www.kaggle.com/datasets/gautamkarat/api-call-sequences

### Setting up Kaggle API credentials

1. Create a Kaggle account if you don't have one.
2. Go to your Kaggle account settings and create a new API token — this downloads a
   `kaggle.json` file.
3. Place that file at `~/.kaggle/kaggle.json` (on Linux/Mac) or
   `C:\Users\<username>\.kaggle\kaggle.json` (on Windows).
4. If running on Kaggle itself, you can instead add your credentials via Kaggle's
   built-in "Secrets" feature, or simply run it as a Kaggle Notebook where dataset
   access is already handled — no token file needed in that case.
5. Make sure to visit both dataset pages above at least once and accept any usage
   terms if prompted.

## How to Run

The entire pipeline is contained in a single notebook cell. Run it from top to bottom.
It executes these steps in order:

1. Download both datasets from Kaggle
2. Build the API-call vocabulary (integer ID → API name mapping)
3. Load, preprocess, and merge both datasets; split into train/validation/test
   (72% / 8% / 20%); apply SMOTE oversampling to the training set only
4. Train the main TEMPEST model (Embedding + stacked LSTM)
5. Train baseline models: Random Forest, SVM (RBF), CNN, Feedforward
6. Run the architecture ablation study (single LSTM, SimpleRNN, no-embedding LSTM,
   feedforward)
7. Run the sequence-order ablation (shuffle API call order and measure AUC drop)
8. Run the 15-seed statistical validation with a Wilcoxon signed-rank test
   (this is the slowest step — it retrains 30 models)
9. Run the truncation-curve experiment (train on the first N calls only, for
   N = 10, 20, ..., 100)
10. Compute Integrated Gradients attributions and map them to API call names
11. Measure inference latency for all models
12. Generate all figures and save them to disk
13. Save all metrics to a results JSON file

Expect the full run to take **30–60+ minutes or more** on a GPU, depending on hardware,
mainly due to step 8.

## Outputs

Running the notebook produces the following files (not just printed output):

- `results/all_results.json` — every metric reported in the paper (AUC, accuracy,
  precision/recall/F1 for all models and test sets, ablation deltas, shuffle-ablation
  deltas, 15-seed statistics, truncation curve, runtime, top-20 attributed positions)
- `results/vocab_map.json` — integer API ID → API name mapping
- `results/api_attribution_findings.json` — top-20 most influential sequence
  positions with resolved API names
- `figures/*.png` — ROC curves, truncation curve, Integrated Gradients plots,
  confusion matrices, training history, statistical comparison boxplot, and
  accuracy comparison bar chart (10+ figures total)

**Note:** figures are saved to disk but not displayed inline in the notebook (there
are no `plt.show()` calls). To view them, either download the `figures/` folder from
the Kaggle output panel, or add `plt.show()` after each `plt.savefig()` call if you
want them to render inline.

## Reproducibility

- The main model, baselines, and ablations all use a fixed seed (`random_state=42` /
  `seed=42`), so results should reproduce closely given the same data split.
- The 15-seed statistical test uses this fixed seed list:
  `[42, 123, 456, 789, 1337, 2024, 314, 999, 7, 2718, 100, 200, 300, 400, 500]`
- Minor numerical differences between runs (e.g., in the fourth decimal place of AUC)
  are expected due to GPU non-determinism in TensorFlow, even with fixed seeds.

## Known Issue to Check

In one archived run of this notebook, the vocabulary-building step failed because the
expected file (`sample_analysis_data.txt`) was not found in the downloaded dataset. When
this happens, the code falls back to generic placeholder names (`api_0`, `api_1`, ...)
instead of real Windows API names. If your run produces placeholder names instead of
real API names (e.g. `NtSetInformationFile`, `CryptAcquireContextW`) in the
Integrated Gradients results, check that the dataset actually contains
`sample_analysis_data.txt` (or the equivalent raw text file the vocabulary step reads
from) after downloading.
