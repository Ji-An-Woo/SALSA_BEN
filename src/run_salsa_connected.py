import os, subprocess, json, re, time
from pathlib import Path
from tqdm import tqdm
from src.utils import ensure_dir, save_json
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

# read datasets list
cfg = json.load(open('configs/light_params.json'))
datasets = cfg['datasets']

def build_cmd(data_path, exp_name, seed):
    flags = cfg.get('salsa_train_flags', {})
    cmd = ['python3', str(EXTERNAL_TRAIN_SCRIPT),
           '--data_path', str(data_path),
           '--exp_name', exp_name,
           '--secret_seed', str(seed),
           '--rlwe', '0',
           '--task', 'mlwe-i',
           '--angular_emb', 'true',
           '--dxdistinguisher', 'true',
           '--hamming', '3']
    # add simple model flags (small for fast runs)
    cmd += ['--train_batch_size', str(flags.get('train_batch_size',32)),
            '--val_batch_size', str(flags.get('val_batch_size',64)),
            '--n_enc_heads', str(flags.get('n_enc_heads',4)),
            '--n_enc_layers', str(flags.get('n_enc_layers',2)),
            '--enc_emb_dim', str(flags.get('enc_emb_dim',128)),
            '--max_epoch', str(flags.get('epochs',5))]
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
