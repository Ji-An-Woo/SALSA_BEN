#!/usr/bin/env python3
"""
SALSA 직접 실행 스크립트
======================
"""
import os
import subprocess
import sys
from pathlib import Path

def run_salsa_direct():
    """SALSA를 직접 실행"""
    
    # 경로 설정
    root_dir = Path(__file__).parent
    lwe_dir = root_dir / "external" / "LWE-benchmarking"
    data_dir = root_dir / "data" / "precomputed" / "baseline_n10"
    salsa_script = lwe_dir / "src" / "salsa" / "train_and_recover.py"
    
    print(f"🔍 경로 확인:")
    print(f"   Root: {root_dir}")
    print(f"   LWE dir: {lwe_dir}")
    print(f"   SALSA script: {salsa_script}")
    print(f"   Data dir: {data_dir}")
    
    print(f"\n📁 파일 존재 확인:")
    print(f"   LWE dir exists: {lwe_dir.exists()}")
    print(f"   SALSA script exists: {salsa_script.exists()}")
    print(f"   Data dir exists: {data_dir.exists()}")
    
    if not all([lwe_dir.exists(), salsa_script.exists(), data_dir.exists()]):
        print("❌ 필요한 파일이나 디렉토리가 없습니다!")
        return False
    
    # 작업 디렉토리 변경
    original_cwd = os.getcwd()
    os.chdir(str(lwe_dir))
    
    print(f"\n🚀 SALSA 실행 중...")
    print(f"작업 디렉토리: {os.getcwd()}")
    
    try:
        # SALSA 명령어 구성
        cmd = [
            sys.executable,  # python 실행 파일
            "src/salsa/train_and_recover.py",
            "--data_path", str(data_dir),
            "--exp_name", "direct_test",
            "--secret_seed", "0",
            "--hamming", "3", 
            "--task", "lwe",
            "--epochs", "1",
            "--train_batch_size", "4",
            "--val_batch_size", "8",
            "--cpu", "false",
            "--dtype", "float16"
        ]
        
        print(f"실행 명령어: {' '.join(cmd)}")
        
        # 실행
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"\n📊 실행 결과:")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"\n📝 stdout (마지막 20줄):")
            lines = result.stdout.strip().split('\n')
            for line in lines[-20:]:
                print(f"   {line}")
        
        if result.stderr:
            print(f"\n❌ stderr:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        return False
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = run_salsa_direct()
    print(f"\n{'✅ 성공!' if success else '❌ 실패!'}")