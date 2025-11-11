#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SALSA-repro 간단 실행 스크립트
============================
"""

import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, desc):
    print(f"\n🔄 {desc}")
    print(f"실행: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, text=True)
        if result.returncode == 0:
            print("✅ 성공!")
            return True
        else:
            print(f"❌ 실패! (코드: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ 에러: {e}")
        return False

def main():
    print("🚀 SALSA-repro 간단 실행")
    print("=" * 40)
    
    # 1. PyTorch 설치
    print("\n📦 PyTorch 설치 중...")
    run_cmd("py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", "CUDA PyTorch 설치")
    
    # 2. requirements 설치
    run_cmd("py -m pip install -r requirements.txt", "기타 패키지 설치")
    
    # 3. 레포 클론 (없으면)
    if not Path("external/LWE-benchmarking").exists():
        run_cmd("git clone https://github.com/facebookresearch/LWE-benchmarking external/LWE-benchmarking", "LWE-benchmarking 클론")
    else:
        print("✅ LWE-benchmarking 이미 존재")
    
    # 4. 데이터 생성
    run_cmd("py src/data_gen_obfuscate_fixed.py", "데이터 생성")
    
    # 5. SALSA 실행
    run_cmd("py src/run_salsa_connected.py", "SALSA 실행")
    
    # 6. 결과 평가
    run_cmd("py src/evaluate_and_plot.py", "결과 평가")
    
    print("\n🎉 완료! 결과는 results/salsa_runs/ 폴더에 있습니다.")

if __name__ == '__main__':
    main()