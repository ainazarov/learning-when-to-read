# Learning When to Read

Cost-aware arXiv paper classification with supervised NLP models and a reinforcement learning policy for selective abstract reading.

## Project Summary

This project studies whether an NLP classifier can predict arXiv primary categories accurately while avoiding unnecessary abstract processing. The work has two stages:

1. Compare supervised classifiers under two input settings: title-only and title-plus-abstract.
2. Train a reinforcement learning policy that decides when the abstract should be read, using confidence features from the title-only model.

The dataset contains 6000 recent arXiv papers from six categories:

- `cs.AI`
- `cs.CL`
- `cs.CV`
- `cs.LG`
- `cs.NE`
- `stat.ML`

## Latest Results

Best supervised classifier:

- SciBERT with title + abstract
- Test accuracy: `0.752`
- Test macro F1: `0.750`

Best title-only classifier:

- SciBERT with title only
- Test accuracy: `0.696`
- Test macro F1: `0.696`

Adaptive RL policy:

- Abstract usage: `15.2%`
- Test accuracy: `0.723`
- Test macro F1: `0.720`
- Average reward: `0.701`
- Abstract reads saved vs always reading abstracts: `84.8%`

## Repository Structure

```text
.
├── main.ipynb                         # Main experiment notebook
├── requirements.txt                   # Python dependencies
├── scripts/
│   ├── download_arxiv_dataset.py      # Resumable arXiv API downloader
│   └── parse_arxiv_articles.ipynb
├── data/
│   ├── raw/                           # Cached raw arXiv data
│   ├── processed/                     # Cleaned dataset
│   └── splits/                        # Train/validation/test splits
├── results/
│   ├── figures/                       # Report figures
│   └── tables/                        # Metrics and diagnostics
└── report/
    ├── main.tex                       # LaTeX report source
    ├── lit.bib                        # Bibliography
    └── main.pdf                       # Compiled report
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The experiments use PyTorch and Hugging Face Transformers. A CUDA GPU is recommended for rerunning transformer fine-tuning.

## Dataset

The cached dataset is already stored under `data/`. If the raw CSV is missing, it can be downloaded again with:

```bash
python scripts/download_arxiv_dataset.py \
  --output data/raw/arxiv_1000_per_category.csv \
  --progress data/raw/arxiv_1000_per_category_progress.json \
  --papers-per-category 1000 \
  --categories cs.CL cs.LG cs.CV cs.AI cs.NE stat.ML
```

The downloader uses the official arXiv API, queries one category at a time, sorts by submitted date, and stores progress after each page.

## Running Experiments

Open and run:

```text
main.ipynb
```

The notebook:

- loads or downloads the arXiv dataset
- preprocesses and stratifies the data
- trains/evaluates TF-IDF + Linear SVM
- fine-tunes SciBERT, DistilBERT, and MiniLM
- trains the RL policy over title-only confidence features
- writes result tables to `results/tables/`
- writes figures to `results/figures/`

## Report

The final report source is in `report/main.tex`.

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

Useful outputs:

- `results/tables/baseline_results.csv`
- `results/tables/rl_results.csv`
- `results/tables/final_summary.md`
- `results/figures/best_model_confusion_matrix.png`
- `results/figures/accuracy_vs_abstract_usage.png`
- `report/main.pdf`
