# %% [markdown]
# # Q3: Adult Income - EDA, baseline, AutoML (TPOT), save model
# Pipeline: EDA -> preprocessing -> Logistic Regression baseline -> TPOT -> compare -> save -> metrics.json
# The saved model + metrics.json are used by api.py (FastAPI) and app.py (Streamlit).

# %% Imports
import json, time, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance

# %% 1. EDA: class balance, feature distributions, correlations ('?' = missing)
df = pd.read_csv("../ASSIGNMENT03/adult.csv").replace("?", np.nan)
print(df.shape, "\nMissing per column:\n", df.isna().sum()[lambda s: s > 0])
print(df["income"].value_counts(normalize=True))            # ~76% <=50K -> imbalanced
df.hist(figsize=(10, 6)); plt.tight_layout(); plt.savefig("results/q3_distributions.png"); plt.show()
print(df.select_dtypes("number").corr().round(2))

# %% 2. Preprocessing + a derived feature (net capital)
df["net.capital"] = df["capital.gain"] - df["capital.loss"]
y = (df["income"].str.strip() == ">50K").astype(int)
X = df.drop(columns="income")
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% 3. Baseline: Logistic Regression
t = time.time()
base = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
base_time = time.time() - t
base_acc, base_f1 = accuracy_score(yte, base.predict(Xte)), f1_score(yte, base.predict(Xte))
print(f"Baseline  acc={base_acc:.4f} f1={base_f1:.4f} time={base_time:.1f}s")

# %% 4. AutoML with TPOT (time-limited to a few minutes)
from tpot import TPOTClassifier
Xtr_p = prep.fit_transform(Xtr)                      # numeric matrix for TPOT
t = time.time()
try:     # TPOT >= 1.0
    tpot = TPOTClassifier(search_space="linear-light", max_time_mins=5, cv=3, n_jobs=4, random_state=42, verbose=1)
except TypeError:   # TPOT 0.x
    tpot = TPOTClassifier(generations=3, population_size=10, max_time_mins=5, cv=3, n_jobs=4, random_state=42, verbosity=2)
tpot.fit(Xtr_p, ytr)
auto_time = time.time() - t
auto = Pipeline([("prep", prep), ("clf", tpot.fitted_pipeline_)])   # raw input -> prediction
auto_acc, auto_f1 = accuracy_score(yte, auto.predict(Xte)), f1_score(yte, auto.predict(Xte))
print(f"TPOT      acc={auto_acc:.4f} f1={auto_f1:.4f} time={auto_time:.1f}s")
print("Best pipeline:", tpot.fitted_pipeline_)

# %% 5. Comparison + 6. Save the best model
print(pd.DataFrame({"Model": ["Logistic Regression", "TPOT"], "Accuracy": [base_acc, auto_acc],
                    "F1": [base_f1, auto_f1], "Train time (s)": [base_time, auto_time]}).round(4))
best, best_acc, name = (auto, auto_acc, "TPOT") if auto_acc >= base_acc else (base, base_acc, "Logistic Regression")
joblib.dump(best, "best_adult_income_model.joblib")

# %% Top-5 features (permutation importance on raw columns) -> metrics.json for the Streamlit app
sub = Xte.sample(1500, random_state=42)
pi = permutation_importance(best, sub, yte.loc[sub.index], n_repeats=3, random_state=42)
top5 = pd.Series(pi.importances_mean, index=Xte.columns).nlargest(5)
print(top5)
json.dump({"model": name, "accuracy": round(best_acc, 4), "f1": round(float(f1_score(yte, best.predict(Xte))), 4),
           "top5": {k: round(float(v), 4) for k, v in top5.items()}}, open("metrics.json", "w"), indent=2)

# %% [markdown]
# **Discussion (fill in from YOUR results):** AutoML searches models/hyperparameters automatically and can gain a
# little accuracy, but costs minutes instead of seconds and yields a more complex, less interpretable model;
# the baseline is fast, simple and explainable. Class imbalance (~76/24) means F1 matters beside accuracy.
