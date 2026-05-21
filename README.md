# Learning When to Read

Reinforcement learning for selective context use in transformer-based scientific text categorization.

This repository accompanies the report **"Learning When to Read: Reinforcement Learning for Selective Context Use in Transformer-Based Scientific Text Categorization"**. The project studies arXiv primary-category classification as a cost-aware two-stage problem:

1. Train supervised classifiers with either titles only or titles plus abstracts.
2. Train a policy-gradient reinforcement learning controller that decides when an abstract is worth reading.

The controller observes confidence features from the title-only model and chooses between keeping the cheap title-only prediction or querying the higher-context title-plus-abstract model. The goal is to retain much of the accuracy benefit of abstracts while avoiding unnecessary long-input processing.

## Headline Results

The dataset contains the most recent arXiv papers collected on May 21, 2026 from six primary categories. The final test set has 900 papers.


| System                      | Abstract use | Test accuracy | Test macro F1 | Notes                         |
| --------------------------- | ------------ | ------------- | ------------- | ----------------------------- |
| SciBERT, title + abstract   | 100.0%       | 0.752         | 0.750         | Best supervised classifier    |
| SciBERT, title only         | 0.0%         | 0.696         | 0.696         | Best title-only classifier    |
| RL selective-reading policy | 15.2%        | 0.723         | 0.720         | Saves 84.8% of abstract reads |


The RL policy reaches an average reward of `0.701` with an abstract-reading cost of `0.15`. Compared with always reading abstracts, it gives up `0.029` accuracy while saving most abstract processing.

The full supervised comparison also includes TF-IDF + Linear SVM, DistilBERT, and MiniLM. Abstracts improve every model, but the gain is not uniform: SciBERT gains `0.054` macro F1 from adding abstracts, while TF-IDF + Linear SVM gains `0.108` and MiniLM gains `0.163`. This supports the selective-reading formulation: abstracts help on some papers, but they are not always necessary.

## Dataset

The dataset is balanced by construction: `1000` papers per category, `6000` papers total.

Target categories:

- `cs.AI` - Artificial Intelligence
- `cs.CL` - Computation and Language
- `cs.CV` - Computer Vision and Pattern Recognition
- `cs.LG` - Machine Learning
- `cs.NE` - Neural and Evolutionary Computing
- `stat.ML` - Machine Learning in Statistics

Each row contains a paper ID, title, abstract, primary category, and publication metadata. The processed data is split with stratification:


| Split      | Papers | Papers per category |
| ---------- | ------ | ------------------- |
| Train      | 4200   | 700                 |
| Validation | 900    | 150                 |
| Test       | 900    | 150                 |


Cached data is included under `data/`, so the notebook can run from local files without calling the arXiv API again.

## Method Summary

The supervised stage compares:

- TF-IDF + Linear SVM with unigram and bigram features
- SciBERT: `allenai/scibert_scivocab_uncased`
- DistilBERT: `distilbert-base-uncased`
- MiniLM: `microsoft/MiniLM-L12-H384-uncased`

Each transformer is fine-tuned twice:

- `title`: title-only input, truncated to 64 tokens
- `title_abstract`: title plus abstract input, truncated to 256 tokens

The RL stage uses SciBERT for both roles because it achieves strongest metrics for the collected dataset. The policy state contains the six title-only class probabilities plus four confidence summaries: top-1 probability, top-2 probability, probability margin, and normalized entropy. The policy chooses whether to use the title-only prediction or pay the read abstract cost and use the title-plus-abstract prediction.

## Repository Structure

```text
.
|-- main.ipynb                         # End-to-end experiment notebook
|-- requirements.txt                   # Python dependencies
|-- scripts/
|   |-- download_arxiv_dataset.py      # Resumable official arXiv API downloader
|   `-- download_arxiv_dataset_test_notebook.ipynb
|-- data/
|   |-- raw/
|   |   |-- arxiv_1000_per_category.csv
|   |   `-- arxiv_1000_per_category_progress.json
|   |-- processed/
|   |   `-- arxiv_processed.csv
|   `-- splits/
|       |-- train.csv
|       |-- validation.csv
|       `-- test.csv
|-- results/
|   |-- checkpoints/                   # Fine-tuned transformer checkpoints
|   |-- figures/                       # Report plots
|   `-- tables/                        # Metrics, diagnostics, and summaries
|-- report/
|   |-- main.tex                       # LaTeX report source
|   |-- lit.bib                        # Bibliography
|   `-- main.pdf                       # Compiled report
|-- LICENSE
`-- README.md
```

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments use PyTorch and Hugging Face Transformers. A CUDA GPU is recommended for rerunning transformer fine-tuning.

## Recreating the Dataset

The cached raw CSV is already available at `data/raw/arxiv_1000_per_category.csv`. If it needs to be recreated, run:

```bash
python scripts/download_arxiv_dataset.py \
  --output data/raw/arxiv_1000_per_category.csv \
  --progress data/raw/arxiv_1000_per_category_progress.json \
  --papers-per-category 1000 \
  --categories cs.CL cs.LG cs.CV cs.AI cs.NE stat.ML
```

The downloader queries one category at a time, sorts by submitted date, keeps only papers whose primary category matches the requested category, and stores progress after each page.

## Running Experiments

Open and run:

```text
main.ipynb
```

The notebook:

- loads cached data or invokes the downloader if needed
- preprocesses the dataset and writes stratified splits
- trains and evaluates the TF-IDF + Linear SVM baseline
- fine-tunes SciBERT, DistilBERT, and MiniLM
- trains the RL selective-reading policy
- writes tables to `results/tables/`
- writes plots to `results/figures/`
- stores transformer checkpoints under `results/checkpoints/`

## Report

The final report source is:

```text
report/main.tex
```

To compile it:

```bash
cd report
latexmk -pdf main.tex
```

The compiled PDF is:

```text
report/main.pdf
```

## Main Artifacts

Key result files:

- `results/tables/baseline_results.csv`
- `results/tables/rl_results.csv`
- `results/tables/final_summary.md`
- `results/tables/abstract_gain_diagnostics.csv`
- `results/tables/threshold_validation_test_diagnostics.csv`
- `results/tables/classification_reports.json`
- `results/figures/best_model_confusion_matrix.png`
- `results/figures/accuracy_vs_abstract_usage.png`
- `results/figures/rl_cost_sweep_accuracy_reward.png`

Model checkpoints are generated under `results/checkpoints/transformers/`.