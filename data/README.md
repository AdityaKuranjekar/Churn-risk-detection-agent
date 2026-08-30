# data/ — where to put the dataset

## 1. Download the Kaggle dataset

https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset

Download the CSV (Kaggle gives you a `.zip` — extract it).

## 2. Place and rename it here

```
data/customer_churn_raw.csv
```

- **Location:** this `data/` folder, at the project root
  (`D:\Professional Work\Business\Product_Space_Agent\data\customer_churn_raw.csv`)
- **Format:** plain `.csv`, UTF-8, with a header row (as downloaded from Kaggle)
- **Exact filename:** `customer_churn_raw.csv` (this is what `DATASET_CSV` in `.env` points to)

If the extracted file has a different name (e.g. `customer_churn_dataset.csv`),
just rename it to `customer_churn_raw.csv`.

## 3. Other files in this folder

| File | Who creates it | Purpose |
|---|---|---|
| `customer_churn_raw.csv` | **you** (download) | raw training + demo data |
| `playbook.json` | generated in build Step 4 | 8–12 retention playbook snippets |
| `playbook_embeddings.pkl` | generated on first app start | cached Ollama embedding vectors |

Only `customer_churn_raw.csv` needs to come from you.
