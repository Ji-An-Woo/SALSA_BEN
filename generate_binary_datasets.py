#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import csv
import json
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

def obfuscate_binary_polynomial(secret, coeffs, degrees):
    """다항식 난독화 (이진 버전)"""
    n = len(secret)
    result = np.zeros(n, dtype=np.int32)
    
    for degree in degrees:
        coeff = coeffs.get(degree, 0)
        if degree == 1:
            result = (result + coeff * secret) % 2
        else:
            # 이진에서 거듭제곱은 자기 자신
            result = (result + coeff * secret) % 2
    
    return result

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

def create_salsa_files(A, b, secret, output_path, hamming=3, seed=0):
    """SALSA 호환 파일들 생성"""
    
    # 기본 파일들
    np.save(f"{output_path}/A.npy", A)
    np.save(f"{output_path}/b.npy", b)
    
    # SALSA train/val/test 분할 (간단히 복사)
    for split in ['train', 'val', 'test', 'orig']:
        np.save(f"{output_path}/{split}_A.npy", A)
        np.save(f"{output_path}/{split}_b.npy", b)
        
        # hamming별 파일들
        for i in range(3):  # 0, 1, 2
            np.save(f"{output_path}/{split}_b_{hamming}_{i}.npy", b)
    
    # 시드별 파일들 (seed=0, seed=42 등)
    for s in [0, 42]:
        for split in ['train', 'val', 'test', 'orig']:
            np.save(f"{output_path}/{split}_b_{hamming}_{s}.npy", b)
    
    # 파라미터 저장
    params = {
        'N': len(secret),
        'Q': 842779,
        'logq': 20,
        'gamma': 3.0,
        'm': len(b),
        'secret_distribution': 'binary',
        'hamming_weight': np.sum(secret)
    }
    
    with open(f"{output_path}/params.pkl", 'wb') as f:
        pickle.dump(params, f)
    
    # 메타데이터 저장
    meta = {
        "s": secret.tolist(),
        "params": {
            "name": f"n{len(secret)}",
            "n": int(len(secret)),
            "q": int(842779),
            "m": int(len(b)),
            "sigma": float(3.0),
            "hamming": int(np.sum(secret)),
            "seed": int(seed)
        }
    }
    
    with open(f"{output_path}/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

def create_binary_dataset(name, n, m, q, sigma, hamming_weight, seed, output_path, 
                         coeffs=None, degrees=None, is_idea=False):
    """이진 비밀키 데이터셋 생성"""
    
    print(f"🔄 {name} 이진 데이터셋 생성 중...")
    
    # 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 이진 비밀키 생성
    secret = sample_binary_secret(n, hamming_weight, seed)
    
    if is_idea and coeffs and degrees:
        # 아이디어: 다항식 변형 적용 (이진)
        secret = obfuscate_binary_polynomial(secret, coeffs, degrees)
    
    print(f"✅ 이진 비밀키: {secret}")
    print(f"   Hamming weight: {np.sum(secret)}")
    
    # LWE 샘플 생성
    A, b = gen_lwe_samples(secret, m, q, sigma, seed)
    
    # SALSA 호환 파일들 생성
    create_salsa_files(A, b, secret, output_path, hamming=3, seed=seed)
    
    print(f"✅ {name} 완료! ({output_path})")
    return secret

def main():
    """모든 데이터셋을 이진으로 재생성"""
    
    print("🎯 모든 데이터셋을 이진 비밀키로 재생성...")
    
    # 기본 파라미터
    q = 842779
    sigma = 3.0
    hamming_weight = 3
    
    # 다항식 계수 (이진 버전)
    coeffs = {1: 1, 3: 1, 5: 1}
    degrees = [1, 3, 5]
    
    datasets = []
    
    # 1. baseline_n10_binary
    secret1 = create_binary_dataset(
        "baseline_n10_binary", 10, 500, q, sigma, hamming_weight, 111,
        "data/precomputed/baseline_n10_binary"
    )
    datasets.append(("baseline_n10_binary", "baseline", 10, 500, secret1))
    
    # 2. idea_n10_binary 
    secret2 = create_binary_dataset(
        "idea_n10_binary", 10, 500, q, sigma, hamming_weight, 111,
        "data/precomputed/idea_n10_binary",
        coeffs=coeffs, degrees=degrees, is_idea=True
    )
    datasets.append(("idea_n10_binary", "idea", 10, 500, secret2))
    
    # 3. baseline_n30_binary
    secret3 = create_binary_dataset(
        "baseline_n30_binary", 30, 2000, q, sigma, hamming_weight, 222,
        "data/precomputed/baseline_n30_binary"
    )
    datasets.append(("baseline_n30_binary", "baseline", 30, 2000, secret3))
    
    # 4. idea_n30_binary
    secret4 = create_binary_dataset(
        "idea_n30_binary", 30, 2000, q, sigma, hamming_weight, 222,
        "data/precomputed/idea_n30_binary",
        coeffs=coeffs, degrees=degrees, is_idea=True
    )
    datasets.append(("idea_n30_binary", "idea", 30, 2000, secret4))
    
    # CSV 업데이트
    print("\n📊 CSV 파일 업데이트...")
    
    with open('data/precomputed/binary_datasets_params.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'type', 'n', 'm', 'secret', 'path'])
        
        for name, dtype, n, m, secret in datasets:
            path = f"C:\\Users\\user\\OneDrive\\Desktop\\salsa-repro-salsa-connected\\data\\precomputed\\{name}"
            writer.writerow([name, dtype, n, m, secret.tolist(), path])
    
    print("🎉 모든 이진 데이터셋 생성 완료!")
    print("📁 생성된 폴더들:")
    for name, _, _, _, _ in datasets:
        print(f"   - data/precomputed/{name}")

if __name__ == "__main__":
    main()