import os, subprocess, json, time
from pathlib import Path

def run_salsa_experiment(data_path, exp_name, epochs=5):
    """SALSA 실험 실행"""
    cmd = [
        'py', 'src/salsa/train_and_recover.py',
        '--data_path', f'../../{data_path}',
        '--exp_name', exp_name,
        '--epochs', str(epochs),
        '--task', 'lwe',
        '--secret_seed', '0',
        '--hamming', '3',
        '--cpu', 'true'  # CPU 모드로 실행
    ]

            print(f"   실행 시간: {duration:.1f}초")

    # working directory를 LWE-benchmarking으로 변경
    original_cwd = os.getcwd()
    env = os.environ.copy()
    env['PYTORCH_JIT'] = '0'  # JIT 컴파일 비활성화
    env['TORCH_USE_CUDA_DSA'] = '1'
    try:
        os.chdir('external/LWE-benchmarking')
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', env=env)
        return result.returncode, result.stdout, result.stderr
    finally:
        os.chdir(original_cwd)

def main():
    print("🎯 SALSA 5 Epochs 실험 실행")

    experiments = [
        ('data/precomputed/baseline_n10', 'baseline_n10_5epochs'),
        ('data/precomputed/idea_n10', 'idea_n10_5epochs')
    ]

    results = []

    for data_path, exp_name in experiments:
        print(f"\n📊 {exp_name} 실행 중...")

        start_time = time.time()
        returncode, stdout, stderr = run_salsa_experiment(data_path, exp_name, epochs=5)
        end_time = time.time()

        duration = end_time - start_time

        # 결과 저장
        result = {
            'experiment': exp_name,
            'data_path': data_path,
            'epochs': 5,
            'returncode': returncode,
            'duration': duration,
            'stdout': stdout[-2000:],  # 마지막 2000자만 저장
            'stderr': stderr[-1000:] if stderr else ''
        }

        results.append(result)

        print(f"   ⏱️ 실행 시간: {duration:.1f}초")
        print(f"   반환 코드: {returncode}")

        # 비밀키 복구 확인
        if 'Best secret guess' in stdout:
            print("   ✅ 비밀키 복구 성공!")
        else:
            print("   ❌ 비밀키 복구 실패")

    # CSV로 결과 저장
    import csv
    with open('salsa_5epochs_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'data_path', 'epochs', 'returncode', 'duration', 'recovery_success'])
        writer.writeheader()

        for result in results:
            recovery_success = 'Best secret guess' in result['stdout']
            writer.writerow({
                'experiment': result['experiment'],
                'data_path': result['data_path'],
                'epochs': result['epochs'],
                'returncode': result['returncode'],
                'duration': result['duration'],
                'recovery_success': recovery_success
            })

    print("\n📊 결과가 salsa_5epochs_results.csv에 저장되었습니다.")

    # 상세 로그 저장
    with open('salsa_5epochs_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("📝 상세 로그가 salsa_5epochs_detailed.json에 저장되었습니다.")

if __name__ == '__main__':
    main()