#!/usr/bin/env bash
# Creates a virtual environment "myenv" in this folder and installs requirements.txt
# Run from the "AI Practice" folder:  bash setup_venv.sh
set -e
# scikit-learn 1.8 / pandas 3 need Python >= 3.11; use the newest one available
for p in python3.13 python3.12 python3.11; do
  if command -v $p >/dev/null 2>&1; then PY=$p; break; fi
done
[ -z "$PY" ] && { echo "Need Python 3.11+ (check: python3 --version). Or use conda: conda create -n myenv python=3.13"; exit 1; }
echo "Using $($PY --version)"
$PY -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name myenv --display-name "myenv"
echo "Done. Activate with: source myenv/bin/activate"
