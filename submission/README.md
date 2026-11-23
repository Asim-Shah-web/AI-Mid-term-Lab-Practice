# Programming for AI - Mid-Term Lab

## Files
| File | Purpose |
|---|---|
| `q1_linear_regression.py` | Q1: regression from scratch vs scikit-learn |
| `q2_random_forest.py` | Q2: Random Forest / Decision Tree / SVM on Heart Disease + Intel Images |
| `q3_automl.py` | Q3: EDA, baseline, TPOT, saves model + `metrics.json` |
| `api.py` | Q3: FastAPI `/predict` |
| `app.py` | Q3: Streamlit UI (calls the API) |

## Reproduce
```bash
bash setup_env.sh && conda activate myenv      # creates env, installs requirements, makes notebooks
# Intel images: download from Kaggle (puneet6060/intel-image-classification), unzip into ../ASSIGNMENT02/
jupyter lab                                    # run q1, q2, q3 notebooks top to bottom (q3 takes ~5 min for TPOT)
uvicorn api:app --reload                       # terminal 1 (run from this folder, after q3 has saved the model)
streamlit run app.py                           # terminal 2
```
Datasets used: `../ASSIGNMENT01`, `../ASSIGNMENT02`, `../ASSIGNMENT03` CSVs.

## Still to do by hand
- Q1: handwritten derivation (scan/photo) and short report.
- Q2: Part A explanation, Breiman summary, GitHub link.
- Q3: deploy `app.py` (+ model, metrics.json, adult.csv, api.py) to Streamlit Cloud / Hugging Face and add the link here.
