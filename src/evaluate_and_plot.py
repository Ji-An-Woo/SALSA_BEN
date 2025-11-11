import json, os, csv, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.utils import ensure_dir
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'salsa_runs'
PLOTS = ROOT / 'results' / 'plots'
ensure_dir(str(PLOTS))

def load_json(p):
    if Path(p).exists():
        return json.load(open(p))
    return None

def compute_recovery(true_s, pred_s):
    true = np.array(true_s)
    pred = np.array(pred_s)
    exact = int(np.array_equal(true, pred))
    bitwise = float((true == pred).sum()) / true.shape[0]
    return exact, bitwise

folders = sorted([p for p in RESULTS.iterdir() if p.is_dir()])
summary = []
for f in tqdm(folders, desc='eval_folders'):
    meta = load_json(f / 'run_meta.json') or {}
    parsed = load_json(f / 'predicted_secrets.json') or {}
    # load ground truth from precomputed data folder
    pre_meta_path = Path(meta.get('folder','')) / 'meta.json'
    true_s = None
    if pre_meta_path.exists():
        pre_meta = load_json(pre_meta_path)
        true_s = pre_meta.get('s', None)
        s_prime = pre_meta.get('s_prime', None)
    # evaluate parsed guesses if any
    if parsed and true_s is not None:
        best = parsed.get('guesses', parsed) if isinstance(parsed, dict) else parsed
        # consider first guess
        guess = best[0] if isinstance(best, list) and len(best)>0 else None
        if guess is not None:
            exact, bitwise = compute_recovery(true_s, guess)
        else:
            exact, bitwise = None, None
    else:
        exact, bitwise = None, None
    # write summary row
    summary.append({'folder': f.name, 'exact_recovery': exact, 'bitwise_recovery': bitwise, 'meta': meta})
# save CSV and JSON
with open(RESULTS / 'salsa_summary.csv','w',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['folder','exact_recovery','bitwise_recovery','meta'])
    writer.writeheader()
    for r in summary:
        writer.writerow(r)
json.dump(summary, open(RESULTS / 'salsa_summary.json','w'), indent=2)
print('Saved salsa_summary.csv and salsa_summary.json; parsed guesses (if any) in each run folder.')
