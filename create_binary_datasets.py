"""
이진 비밀키 데이터셋 생성 스크립트
SALSA 표준에 맞는 {0, 1} 이진 비밀키 사용
"""

import numpy as np
import pickle
import os
import json
import csv
from pathlib import Path

def sample_binary_secret(n, hamming_weight=3, seed=None):
    """이진 비밀키 생성 {0, 1}"""
    if seed is not None:
        np.random.seed(seed)
    
    secret = np.zeros(n, dtype=np.int32)
    # hamming_weight 개의 위치를 1로 설정
    positions = np.random.choice(n, hamming_weight, replace=False)
    secret[positions] = 1
    return secret

def obfuscate_maclaurin_binary(secret, coeffs, degrees):
    """다항식 난독화 (이진 버전)"""
    n = len(secret)
    result = np.zeros_like(secret)
    
    for degree in degrees:
        coeff = coeffs.get(degree, 0)
        if degree == 1:
            result += coeff * secret
        else:
            result += coeff * (secret ** degree)
    
    return result % 2  # 이진으로 유지

def gen_lwe_samples(secret, m, q, sigma, seed=None):
    """LWE 샘플 생성"""
    if seed is not None:
        np.random.seed(seed + 1000)  # 다른 시드 사용
    
    n = len(secret)
    A = np.random.randint(0, q, size=(m, n), dtype=np.int32)
    
    # 노이즈 생성
    noise = np.random.normal(0, sigma, m)
    noise = np.round(noise).astype(np.int32)
    
    # b = A*s + e (mod q)  
    b = (A @ secret + noise) % q
    
    return A.astype(np.int32), b.astype(np.int32)

def create_binary_dataset(name, n, m, q, sigma, hamming_weight, seed, output_path, 
                         coeffs=None, degrees=None, is_idea=False):
    """이진 비밀키 데이터셋 생성"""
    
    print(f"🔄 {name} 이진 데이터셋 생성 중...")
    
    # 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 이진 비밀키 생성
    secret = sample_binary_secret(n, hamming_weight, seed)
    
    if is_idea and coeffs and degrees:
        # 아이디어: 다항식 변형 적용
        secret = obfuscate_maclaurin_binary(secret, coeffs, degrees)
    
    print(f"✅ 이진 비밀키: {secret}")
    print(f"   Hamming weight: {np.sum(secret != 0)}")
    
    # LWE 샘플 생성
    A, b = gen_lwe_samples(secret, m, q, sigma, seed)
    
    # 파일 저장
    np.save(f"{output_path}/A.npy", A)
    np.save(f"{output_path}/b.npy", b)
    
    # SALSA 호환 파일들 생성
    salsa_files = [
        "train_A.npy", "val_A.npy", "test_A.npy", "orig_A.npy",
        "train_b.npy", "val_b.npy", "test_b.npy", "orig_b.npy"
    ]
    
    # 햄밍 웨이트별 파일들
    for hamming in [0, 1, 2, 42]:  # seed 0, 42 등 지원
        salsa_files.extend([
            f"train_b_3_{hamming}.npy", 
            f"val_b_3_{hamming}.npy", 
            f"test_b_3_{hamming}.npy", 
            f"orig_b_3_{hamming}.npy"
        ])
    
    for file in salsa_files:
        if "A" in file:
            np.save(f"{output_path}/{file}", A)
        else:
            np.save(f"{output_path}/{file}", b)
    
    # 메타데이터 저장
    meta = {
        "s": secret.tolist(),
        "params": {
            "name": name,
            "n": n,
            "q": q,
            "m": m,
            "sigma": sigma,
            "hamming": hamming_weight,
            "seed": seed,
            "secret_type": "binary"
        }
    }
    
    with open(f"{output_path}/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    # 파라미터 저장 (SALSA 호환)
    params = {
        'N': n,
        'Q': q, 
        'logq': int(np.log2(q)),
        'gamma': sigma,
        'm': m,
        'secret_distribution': 'binary',
        'hamming_weight': hamming_weight
    }
    
    with open(f"{output_path}/params.pkl", 'wb') as f:
        pickle.dump(params, f)
    
    print(f"✅ {name} 완료!")
    return secret

def main():
    """모든 데이터셋을 이진으로 재생성"""
    
    print("🎯 모든 데이터셋을 이진 비밀키로 재생성...")
    
    # 기본 파라미터
    q = 842779
    sigma = 3.0
    hamming_weight = 3
    
    # 다항식 계수 (이진용)
    coeffs = {1: 1, 3: 1, 5: 1}
    degrees = [1, 3, 5]
    
    datasets = []
    
    # 1. baseline_n10 (이진)
    secret1 = create_binary_dataset(
        "baseline_n10_binary", 10, 500, q, sigma, hamming_weight, 111,
        "data/precomputed/baseline_n10_binary", 
        is_idea=False
    )
    datasets.append(["", "", 500, 10, "n10", 
                    "C:\\Users\\user\\OneDrive\\Desktop\\salsa-repro-salsa-connected\\data\\precomputed\\baseline_n10_binary", 
                    "baseline"])
    
    # 2. idea_n10 (이진 + 다항식)
    secret2 = create_binary_dataset(
        "idea_n10_binary", 10, 500, q, sigma, hamming_weight, 111,
        "data/precomputed/idea_n10_binary",
        coeffs=coeffs, degrees=degrees, is_idea=True
    )
    datasets.append([str(coeffs), str(degrees), 500, 10, "n10",
                    "C:\\Users\\user\\OneDrive\\Desktop\\salsa-repro-salsa-connected\\data\\precomputed\\idea_n10_binary", 
                    "idea"])
    
    # 3. baseline_n30 (이진)
    secret3 = create_binary_dataset(
        "baseline_n30_binary", 30, 2000, q, sigma, hamming_weight, 222,
        "data/precomputed/baseline_n30_binary",
        is_idea=False
    )
    datasets.append(["", "", 2000, 30, "n30",
                    "C:\\Users\\user\\OneDrive\\Desktop\\salsa-repro-salsa-connected\\data\\precomputed\\baseline_n30_binary", 
                    "baseline"])
    
    # 4. idea_n30 (이진 + 다항식)
    secret4 = create_binary_dataset(
        "idea_n30_binary", 30, 2000, q, sigma, hamming_weight, 222,
        "data/precomputed/idea_n30_binary",
        coeffs=coeffs, degrees=degrees, is_idea=True
    )
    datasets.append([str(coeffs), str(degrees), 2000, 30, "n30",
                    "C:\\Users\\user\\OneDrive\\Desktop\\salsa-repro-salsa-connected\\data\\precomputed\\idea_n30_binary", 
                    "idea"])
    
    # CSV 파일 업데이트
    csv_path = "data/precomputed/binary_datasets_params.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["coeffs", "degrees", "m", "n", "name", "path", "type"])
        writer.writerows(datasets)
    
    print(f"\n🎉 모든 이진 데이터셋 생성 완료!")
    print(f"📊 CSV 저장: {csv_path}")
    print("\n📋 생성된 데이터셋:")
    for i, (name, secret) in enumerate([
        ("baseline_n10_binary", secret1),
        ("idea_n10_binary", secret2), 
        ("baseline_n30_binary", secret3),
        ("idea_n30_binary", secret4)
    ], 1):
        print(f"   {i}. {name}: {secret}")
    
    return datasets

if __name__ == '__main__':
    main()