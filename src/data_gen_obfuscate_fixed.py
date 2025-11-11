import numpy as np
from scipy.signal import fftconvolve
import os, json, csv
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'precomputed'
OUT.mkdir(parents=True, exist_ok=True)

def sample_secret(n, hamming=3, seed=None):
    rng = np.random.RandomState(seed)
    s = np.zeros(n, dtype=np.int64)
    ones = rng.choice(n, size=hamming, replace=False)
    s[ones] = 1
    return s

def circular_wrap(conv_result, n, q):
    a = conv_result[:n]
    b = conv_result[n:2*n] if conv_result.shape[0] >= 2*n else np.zeros(n, dtype=np.int64)
    return (a + b) % q

def obfuscate_maclaurin(s, q, degrees=[1,3,5], coeffs=None, coeff_choices=[-1,1]):
    n = len(s)
    rng = np.random.RandomState(0)
    if coeffs is None:
        coeffs = {d: int(rng.choice(coeff_choices)) for d in degrees}
    s = s.astype(np.int64)
    s_prime = np.zeros(n, dtype=np.int64)
    for d in degrees:
        if d == 1:
            term = s.copy()
        else:
            term = s.copy()
            for _ in range(d-1):
                conv = fftconvolve(term, s, mode='full').astype(np.int64)
                term = circular_wrap(conv, n, 842779)
        s_prime = (s_prime + coeffs[d] * term) % 842779
    return s_prime % 842779, coeffs

def gen_lwe_samples(n, q, m, sigma, s, seed=None):
    rng = np.random.RandomState(seed)
    A = rng.randint(low=0, high=q, size=(m,n), dtype=np.int64)
    e = np.round(rng.normal(loc=0.0, scale=sigma, size=(m,))).astype(np.int64) % q
    b = (A.dot(s) + e) % q
    return A, b, e

def save_npy(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, obj)

def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = sorted(rows[0].keys())
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    cfg = json.load(open('configs/light_params.json'))
    rows = []
    print('Generating precomputed datasets (n=10 and n=30)...')
    for ds in tqdm(cfg['datasets'], desc='datasets'):
        name = ds['name']
        n = ds['n']; q = ds['q']; m = ds['m']; sigma = ds['sigma']; hamming = ds['hamming']; seed = ds['seed']
        s = sample_secret(n, hamming=hamming, seed=seed)
        A,b,e = gen_lwe_samples(n,q,m,sigma,s,seed=seed+1)
        outdir = OUT / f'baseline_{name}'
        outdir.mkdir(parents=True, exist_ok=True)
        save_npy(A, outdir / 'A.npy'); save_npy(b, outdir / 'b.npy'); save_npy(e, outdir / 'e.npy')
        json.dump({'s': s.tolist(), 'params': ds}, open(outdir / 'meta.json','w'), indent=2)
        rows.append({'type':'baseline','name':name,'n':n,'m':m,'path':str(outdir)})

        s_prime, coeffs = obfuscate_maclaurin(s, q, degrees=cfg['idea_params']['degrees'], coeff_choices=cfg['idea_params']['coeff_choices'])
        A2,b2,e2 = gen_lwe_samples(n,q,m,sigma,s_prime,seed=seed+2)
        outdir2 = OUT / f'idea_{name}'
        outdir2.mkdir(parents=True, exist_ok=True)
        save_npy(A2, outdir2 / 'A.npy'); save_npy(b2, outdir2 / 'b.npy'); save_npy(e2, outdir2 / 'e.npy')
        json.dump({'s': s.tolist(), 's_prime': s_prime.tolist(), 'coeffs': coeffs, 'params': ds}, open(outdir2 / 'meta.json','w'), indent=2)
        rows.append({'type':'idea','name':name,'n':n,'m':m,'degrees':str(cfg['idea_params']['degrees']),'coeffs':str(coeffs),'path':str(outdir2)})

    write_csv(rows, OUT / 'generated_datasets_params.csv')
    print('Saved precomputed datasets in', OUT)

if __name__ == '__main__':
    main()
