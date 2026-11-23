# Programming for AI: Mid-Term Lab

Course: Programming for AI | Instructor: Dr. Muhammad Sajjad | Exam: Mid-Term Lab (open book)

This repository contains the solutions to the three mid-term questions: linear regression from scratch, Random Forest experiments, and an end-to-end ML project (AutoML, API, web app).

## Repository structure

```
AI Practice/
├── ASSIGNMENT01/              # Q1 dataset: multiple_linear_regression_dataset.csv
├── ASSIGNMENT02/              # Q2 datasets: heart_cleveland_upload.csv (+ Intel images, see below)
├── ASSIGNMENT03/              # Q3 dataset: adult.csv
├── submission/                # Code for all three questions
│   ├── q1_linear_regression.py    -> notebook (Q1)
│   ├── q2_random_forest.py        -> notebook (Q2)
│   ├── q3_automl.py               -> notebook (Q3)
│   ├── api.py                     # FastAPI service (Q3)
│   ├── app.py                     # Streamlit UI (Q3)
│   └── results/                   # Plots saved by the notebooks
├── Archive_v1/                # Earlier versions of the files (kept for reference)
├── requirements.txt
├── setup_venv.sh              # Creates the "myenv" virtual environment
├── mid-term-BS.pdf            # Exam paper
└── randomForest.pdf           # Breiman (2001), "Random Forests"
```

## What each question covers

| Question | Content | Main output |
|---|---|---|
| Q1 | Multiple linear regression with feature scaling, regularized cost and gradient descent, written from scratch and compared with scikit-learn's `LinearRegression` | Convergence curve, final θ, test MSE of both models |
| Q2 | Random Forest vs Decision Tree vs SVM on Heart Disease (tabular) and Intel Image Classification (image); accuracy vs `n_estimators` | Accuracy plots, accuracy and training-time tables |
| Q3 | Adult Income: EDA, Logistic Regression baseline, TPOT AutoML, model saving, FastAPI `/predict`, Streamlit UI | `best_adult_income_model.joblib`, `metrics.json`, running API and UI |

## Setup

Requires Python 3.11 or newer (developed on 3.13).

```bash
cd "AI Practice"
bash setup_venv.sh             # creates ./myenv and installs requirements.txt
source myenv/bin/activate
```

With Anaconda instead:

```bash
conda create -n myenv python=3.13 -y
conda activate myenv
pip install -r requirements.txt
```

If TPOT fails with `No module named 'pkg_resources'`, run `pip install "setuptools<81"`.

## Datasets

| Dataset | Used in | Location | Source |
|---|---|---|---|
| Multiple linear regression (20 rows) | Q1 | `ASSIGNMENT01/` | Provided with the exam |
| Heart Disease (Cleveland, 297 rows) | Q2 | `ASSIGNMENT02/` | UCI Heart Disease |
| Intel Image Classification | Q2 | `ASSIGNMENT02/seg_train/seg_train/` | Kaggle: `puneet6060/intel-image-classification` (about 350 MB, free Kaggle login) |
| Adult Income (32,561 rows) | Q3 | `ASSIGNMENT03/` | UCI Adult |

The Intel images are not included in the repository. Download the zip from Kaggle and extract it into `ASSIGNMENT02/` so that the class folders (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`) sit under `seg_train/seg_train/`. Only `seg_train` is used.

## Reproducing the experiments

All commands are run from the `submission/` folder, because the scripts use relative paths to the datasets.

```bash
cd submission
jupytext --to notebook q1_linear_regression.py q2_random_forest.py q3_automl.py
jupyter lab
```

Select the `myenv` kernel and run each notebook from top to bottom.

| Notebook | Approx. time | Notes |
|---|---|---|
| `q1_linear_regression` | seconds | Saves `results/q1_convergence.png` |
| `q2_random_forest` | 1–2 min | Needs the Intel images; saves plots to `results/` |
| `q3_automl` | about 6–7 min | TPOT is limited to 5 minutes; saves the model and `metrics.json` |

Run `q3_automl` before starting the API or the UI.

## Running the API and web app (Q3)

Open two terminals with `myenv` activated, both in `submission/`:

```bash
python -m uvicorn api:app --reload      # API docs: http://127.0.0.1:8000/docs
python -m streamlit run app.py          # UI: http://localhost:8501
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors",
  "education.num": 13, "marital.status": "Never-married", "occupation": "Adm-clerical",
  "relationship": "Not-in-family", "race": "White", "sex": "Male",
  "capital.gain": 2174, "capital.loss": 0, "hours.per.week": 40, "native.country": "United-States"
}'
```

The response has the form `{"prediction": "<=50K"}`. The Streamlit sidebar shows the test accuracy and the top 5 most important features, both read from `metrics.json`.

## Methods in brief

- **Q1.** Features are standardized (fit on the training set only). The cost is J(θ) = 1/(2m)·[Σ(hθ(x) − y)² + λΣθⱼ²] with j ≥ 1. The bias term θ₀ is not regularized. The parameters are updated by gradient descent.
- **Q2.** Random Forests work by combining strong individual trees with low correlation between them (bagging plus random feature subsets), as described by Breiman (2001). Images are resized to 32×32, flattened, and subsampled to 250 per class to keep the SVM fast.
- **Q3.** The preprocessing pipeline imputes missing values, one-hot encodes categorical columns and scales numeric ones, and adds a derived `net.capital` feature. Model comparison uses accuracy, F1 and training time. Top-5 feature importances use permutation importance on the raw input columns, so they work for any model TPOT selects.

## Results

Results are produced by running the notebooks and are saved in `submission/results/` and `submission/metrics.json`. Fill in this table from your own run:

| Experiment | Result |
|---|---|
| Q1: test MSE (gradient descent / scikit-learn) | |
| Q2: Heart Disease, best test accuracy (DT / RF / SVM) | |
| Q2: Intel Images, best test accuracy (DT / RF / SVM) | |
| Q3: baseline accuracy / F1 | |
| Q3: TPOT accuracy / F1 | |

## Links

- GitHub repository: _add link_
- Deployed Streamlit app (Streamlit Cloud or Hugging Face Spaces): _add link_

## References

- L. Breiman, "Random Forests", *Machine Learning*, 45(1), 5–32, 2001 (`randomForest.pdf`).
