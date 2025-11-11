#!/usr/bin/env python3
"""
SALSA-repro 완전 자동화 실행 스크립트
====================================

이 스크립트는 다음 과정을 자동으로 실행합니다:
1. GPU/CUDA 설정 및 확인
2. 필요한 패키지 설치
3. 데이터셋 생성
4. SALSA 실행
5. 결과 평가 및 리포트 생성
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def print_step(step_num, total_steps, description):
    """단계별 진행상황 출력"""
    print(f"\n{'='*60}")
    print(f"🚀 STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*60}")

def run_command(cmd, description, check_result=True):
    """명령어 실행 및 결과 확인"""
    print(f"\n🔄 {description}")
    print(f"실행 명령: {cmd}")
    
    try:
        if isinstance(cmd, str):
            # Windows에서 python -> py로 변경
            if cmd.startswith('python '):
                cmd = cmd.replace('python ', 'py ', 1)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
        if result.returncode == 0:
            print(f"✅ 성공!")
            if result.stdout.strip():
                # 출력이 너무 길면 마지막 10줄만 표시
                lines = result.stdout.strip().split('\n')
                if len(lines) > 10:
                    print("...")
                    print('\n'.join(lines[-10:]))
                else:
                    print(result.stdout.strip())
            return True
        else:
            print(f"❌ 실패!")
            if result.stderr.strip():
                print(f"에러: {result.stderr.strip()}")
            if check_result:
                return False
            return True
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        if check_result:
            return False
        return True

def check_and_install_cuda():
    """CUDA 환경 체크 및 PyTorch 설치"""
    print("🔍 NVIDIA GPU 확인 중...")
    
    # NVIDIA 드라이버 체크
    nvidia_ok = run_command("nvidia-smi", "NVIDIA 드라이버 확인", check_result=False)
    
    if nvidia_ok:
        print("✅ NVIDIA GPU 감지됨 - CUDA 버전 PyTorch 설치")
        # CUDA 버전 PyTorch 설치
        success = run_command(
            "py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "CUDA PyTorch 설치"
        )
        if not success:
            print("⚠️ CUDA PyTorch 설치 실패, CPU 버전으로 대체")
            run_command("py -m pip install torch torchvision torchaudio", "CPU PyTorch 설치")
    else:
        print("⚠️ NVIDIA GPU 미감지 - CPU 버전 PyTorch 설치")
        run_command("py -m pip install torch torchvision torchaudio", "CPU PyTorch 설치")
    
    # CUDA 사용 가능 여부 테스트
    test_cuda_script = '''# -*- coding: utf-8 -*-
import torch
print(f"PyTorch 버전: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA 사용 가능! GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA 사용 불가 - CPU 모드로 실행됩니다")
'''
    
    with open('test_cuda_temp.py', 'w', encoding='utf-8') as f:
        f.write(test_cuda_script)
    
    run_command("py test_cuda_temp.py", "CUDA 사용 가능 여부 테스트")
    os.remove('test_cuda_temp.py')

def main():
    print("🎯 SALSA-repro 완전 자동화 실행 시작!")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: GPU 설정 및 패키지 설치
    print_step(1, 5, "GPU 설정 및 필수 패키지 설치")
    check_and_install_cuda()
    
    # 나머지 requirements 설치
    if not run_command("py -m pip install -r requirements.txt", "나머지 필수 패키지 설치"):
        print("❌ 패키지 설치 실패")
        return False
    
    # Step 2: LWE-benchmarking 레포 확인
    print_step(2, 5, "LWE-benchmarking 레포지토리 확인")
    external_path = Path("external/LWE-benchmarking")
    
    if not external_path.exists():
        print("📥 LWE-benchmarking 레포지토리 클론 중...")
        if not run_command(
            "git clone https://github.com/facebookresearch/LWE-benchmarking external/LWE-benchmarking",
            "LWE-benchmarking 레포 클론"
        ):
            print("❌ 레포 클론 실패")
            return False
    else:
        print("✅ LWE-benchmarking 레포지토리 이미 존재")
    
    # Step 3: 데이터셋 생성
    print_step(3, 5, "LWE 데이터셋 생성")
    if not run_command("py src/data_gen_obfuscate_fixed.py", "데이터셋 생성 (n=10, n=30)"):
        print("❌ 데이터셋 생성 실패")
        return False
    
    # Step 4: SALSA 훈련 실행
    print_step(4, 5, "SALSA 훈련 실행")
    print("⏰ 이 단계는 GPU 성능에 따라 몇 분~몇십 분 소요될 수 있습니다...")
    if not run_command("py src/run_salsa_connected.py", "SALSA 훈련 및 비밀키 복구"):
        print("❌ SALSA 훈련 실패")
        return False
    
    # Step 5: 결과 평가
    print_step(5, 5, "결과 평가 및 리포트 생성")
    if not run_command("py src/evaluate_and_plot.py", "결과 평가 및 요약"):
        print("❌ 결과 평가 실패")
        return False
    
    # 완료 요약
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "🎉" * 20)
    print("🎉 SALSA-repro 실행 완료! 🎉")
    print("🎉" * 20)
    print(f"⏱️  총 실행 시간: {elapsed/60:.1f}분")
    print("\n📊 결과 확인:")
    print("   - 상세 로그: results/salsa_runs/")
    print("   - 요약 결과: results/salsa_runs/salsa_summary.csv")
    print("   - JSON 결과: results/salsa_runs/salsa_summary.json")
    
    # 결과 미리보기
    summary_file = Path("results/salsa_runs/salsa_summary.csv")
    if summary_file.exists():
        print("\n📋 결과 미리보기:")
        try:
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:5]):  # 처음 5줄만
                    print(f"   {line.strip()}")
                if len(lines) > 5:
                    print(f"   ... (총 {len(lines)} 줄)")
        except:
            print("   결과 파일 읽기 실패")
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print("\n✅ 모든 작업이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n❌ 일부 작업이 실패했습니다.")
        sys.exit(1)