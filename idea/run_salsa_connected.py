import os, subprocess, json, re, time
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ensure_dir, save_json
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'external' / 'LWE-benchmarking'   # must clone here
RESULTS = ROOT / 'results' / 'salsa_runs'
ensure_dir(str(RESULTS))

# Update this path if the external repo layout differs
EXTERNAL_TRAIN_SCRIPT = EXTERNAL / 'src' / 'salsa' / 'train_and_recover.py' 
# fallback path based on README example
if not EXTERNAL_TRAIN_SCRIPT.exists():
    EXTERNAL_TRAIN_SCRIPT = EXTERNAL / 'src' / 'salsa' / 'train_and_recover.py'

if not EXTERNAL_TRAIN_SCRIPT.exists():
    print('ERROR: Could not find train_and_recover.py in external/LWE-benchmarking repo. Please clone the repo and check path.')
    print('Expected at:', EXTERNAL_TRAIN_SCRIPT)
    raise SystemExit(1)

# dataset configs (hardcoded)
datasets = [
    {'name': 'n10', 'n': 10, 'q': 842779, 'm': 500, 'sigma': 3.0, 'hamming': 3, 'seed': 111},
    {'name': 'n30', 'n': 30, 'q': 842779, 'm': 2000, 'sigma': 3.0, 'hamming': 3, 'seed': 222}
]

def build_cmd(data_path, exp_name, seed):
    flags = {}
    # Use the venv python directly
    python_exe = str(ROOT / '.venv' / 'bin' / 'python3')
    dump_path = ROOT / 'results' / 'salsa_dumps'
    dump_path.mkdir(parents=True, exist_ok=True)
    # Build command to run from EXTERNAL directory so src is importable
    cmd = [
        'bash', '-c',
        f'cd {EXTERNAL} && {python_exe} src/salsa/train_and_recover.py ' +
        f'--data_path {data_path} ' +
        f'--exp_name {exp_name} ' +
        f'--secret_seed {seed} ' +
        f'--dump_path {dump_path} ' +
        f'--rlwe 0 ' +
        f'--task mlwe-i ' +
        f'--angular_emb true ' +
        f'--dxdistinguisher true ' +
        f'--hamming 3 ' +
        f'--train_batch_size 32 ' +
        f'--val_batch_size 64 ' +
        f'--n_enc_heads 4 ' +
        f'--n_enc_layers 2 ' +
        f'--enc_emb_dim 512 ' +
        f'--epochs 5'
    ]
    return cmd

# find precomputed dataset folders
pre_dir = ROOT / 'data' / 'precomputed'
folders = sorted([p for p in pre_dir.iterdir() if p.is_dir()])

# run SALSA on each folder containing baseline_*/idea_*
for f in tqdm(folders, desc='salsa_folders'):
    # set experiment name
    exp_name = f.name + '_exp'
    out_dir = RESULTS / f.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(f, exp_name, seed=0)
    print('Running SALSA with command:', ' '.join(cmd))
    # run and capture stdout/stderr
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    stdout_lines = []
    for line in p.stdout:
        print(line, end='')
        stdout_lines.append(line)
    ret = p.wait()
    save_json(out_dir / 'run_stdout.json', {'returncode': ret, 'stdout': stdout_lines})
    # try to parse predicted secret(s) from stdout (simple regex)
    joined = ''.join(stdout_lines)
    # look for "Best secret guess" patterns and extract numbers in brackets
    guesses = re.findall(r'Best secret guess[^\[]*\[([^\]]+)\]', joined)
    parsed = []
    for g in guesses:
        # split numbers, convert to ints
        parts = re.findall(r'-?\d+', g)
        parsed.append([int(x) for x in parts])
    if parsed:
        save_json(out_dir / 'predicted_secrets.json', {'guesses': parsed})
    else:
        # no parsed guesses; save raw stdout for manual inspection
        save_json(out_dir / 'parsed_guess_error.json', {'note': 'no guesses parsed', 'inspect_stdout': str(out_dir / 'run_stdout.json')})
    # also save a small metadata file
    save_json(out_dir / 'run_meta.json', {'folder': str(f), 'cmd': ' '.join(cmd), 'returncode': ret})

print('All SALSA runs completed. Results under results/salsa_runs/')
