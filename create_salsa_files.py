#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SALSA 호환 파일 생성기
===================
"""

import numpy as np
import shutil
from pathlib import Path

def create_salsa_compatible_files():
    """SALSA가 필요로 하는 모든 파일들 생성"""
    
    data_dir = Path("data/precomputed")
    folders = [p for p in data_dir.iterdir() if p.is_dir()]
    
    print(f"🔧 {len(folders)}개 폴더에 SALSA 호환 파일들 생성 중...")
    
    for folder in folders:
        print(f"   📁 {folder.name} 처리 중...")
        
        # 기본 파일들 확인
        A_file = folder / "A.npy"
        b_file = folder / "b.npy"
        
        if not A_file.exists() or not b_file.exists():
            print(f"      ❌ 기본 파일 누락: {folder.name}")
            continue
        
        # A와 b 로드
        A = np.load(A_file)
        b = np.load(b_file)
        
        print(f"      데이터 크기: A={A.shape}, b={b.shape}")
        
        # SALSA가 필요로 하는 다양한 파일들 생성
        file_patterns = [
            "test_A.npy", "test_b.npy",
            "train_A.npy", "train_b.npy", 
            "val_A.npy", "val_b.npy",
            # 해밍 가중치별 파일들
            "test_b_3_0.npy", "test_b_3_1.npy", "test_b_3_2.npy",
            "train_b_3_0.npy", "train_b_3_1.npy", "train_b_3_2.npy",
            "val_b_3_0.npy", "val_b_3_1.npy", "val_b_3_2.npy"
        ]
        
        for pattern in file_patterns:
            target_file = folder / pattern
            if not target_file.exists():
                if "_A.npy" in pattern:
                    np.save(target_file, A)
                elif "_b" in pattern:
                    np.save(target_file, b)
                print(f"         ✅ {pattern} 생성됨")
        
        print(f"      ✅ {folder.name} 완료")
    
    print("🎉 모든 SALSA 호환 파일 생성 완료!")
    
    # 루트 레벨 파일들도 생성
    root_files = ["test_A.npy", "test_b.npy", "train_A.npy", "train_b.npy"]
    
    # baseline_n10을 기준으로 루트 파일들 생성
    baseline_folder = data_dir / "baseline_n10"
    if baseline_folder.exists():
        for root_file in root_files:
            target_path = data_dir / root_file
            source_path = baseline_folder / root_file
            if source_path.exists() and not target_path.exists():
                shutil.copy(source_path, target_path)
                print(f"   📋 루트 레벨 {root_file} 생성됨")

if __name__ == '__main__':
    create_salsa_compatible_files()