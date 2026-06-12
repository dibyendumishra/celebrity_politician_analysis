# Voting With The Stars Paper Code


This repository constitutes the code accompanying the paper [Voting with the Stars: Analyzing Partisan Engagement between Celebrities and Politicians in India](https://dl.acm.org/doi/10.1145/3512981) published at CSCW 2022

---

## Project Structure

```
celebrity_politician_analysis/
├── configs/
│   └── config.yaml              # Centralised paths and parameters
├── data/                        # Raw data (not tracked by git)
├── outputs/                     # Generated plots and exports
│   ├── interactions/
│   ├── topical_models/
│   ├── plots/
│   └── texts/
├── notebooks/
│   ├── 01_interactions.ipynb    # Celebrity ↔ politician interaction extraction
│   ├── 02_topic_modelling.ipynb # BTM topic modelling
│   ├── 03_embeddings.ipynb      # TF-IDF + GloVe embedding clustering
│   └── 04_analysis.ipynb        # Visualisation and final analysis
├── src/
│   ├── data/loaders.py          # Data loading helpers
│   ├── interactions/extractor.py  # Interaction extraction logic
│   ├── topical_models/btm_model.py
│   ├── embeddings/vectorizer.py
│   └── visualization/plots.py
├── tests/
│   └── test_extractor.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/celebrity-politician-analysis.git
cd celebrity-politician-analysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Place your data

Put raw data files under `data/` following the layout in `configs/config.yaml`. This directory is excluded from git.

---

## Configuration

All paths and hyperparameters live in `configs/config.yaml`. Edit this file before running any notebook — you should not need to touch source files for routine experiments.

---

## Running the Pipeline

Run notebooks **in order**:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_interactions.ipynb` | Extract celebrity ↔ politician interactions |
| 2 | `02_topic_modelling.ipynb` | Train BTM topic models per party / group |
| 3 | `03_embeddings.ipynb` | Compute embeddings and cluster tweets |
| 4 | `04_analysis.ipynb` | Produce all plots and summary stats |

---

## Data Requirements

| File | Description |
|------|-------------|
| `data/tweets_drive/*.json` | Per-user tweet JSON files |
| `data/politicians_draft1.csv` | Politician metadata |
| `data/celeb_pol_engagement_df.xlsx` | Celebrity engagement summary |
| `data/200823_patriarchy_final_celebs.csv` | Celebrity category labels |

---

## Reproducibility

- Python **3.8+**
- Random seed fixed in `configs/config.yaml` (`random_seed: 42`)
- Pinned dependency versions in `requirements.txt`
- Pre-trained embeddings: `glove-twitter-50` (auto-downloaded via `gensim.downloader`)

---

## Tests

```bash
pytest tests/
```

---

## License

MIT
