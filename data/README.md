# Data

This project uses the **Algonauts 2021 Mini-Track** dataset.

## Access

Data is not included in this repository per the dataset's terms of use
(non-commercial, non-redistributable).

1. Request access via the Algonauts Project: http://algonauts.csail.mit.edu/challenge.html
2. Set your download link in `.env` as `DROPBOX_DATASET_LINK`
3. Run `notebooks/01_setup_and_data_loading.ipynb` to download and extract into `data/raw/`

## Structure once downloaded

- `data/raw/participants_data/` — ROI pickle files (V1.pkl, FFA.pkl, STS.pkl, etc.)
- `data/raw/videos/` — 1000 stimulus video clips