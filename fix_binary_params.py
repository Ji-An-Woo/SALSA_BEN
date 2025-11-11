import pickle
import numpy as np

def fix_binary_params():
    """이진 데이터셋의 params.pkl을 SALSA와 호환되도록 수정"""
    
    datasets = ['baseline_n10_binary', 'idea_n10_binary', 'baseline_n30_binary', 'idea_n30_binary']
    
    for dataset in datasets:
        params_path = f'data/precomputed/{dataset}/params.pkl'
        
        try:
            # 기존 params 로드
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            
            print(f"🔧 수정 중: {dataset}")
            print(f"   기존 keys: {list(params.keys())}")
            
            # SALSA 호환성을 위한 키 수정/추가
            if 'gamma' in params and 'sigma' not in params:
                params['sigma'] = params['gamma']  # gamma를 sigma로 복사
                print(f"   ✅ sigma 추가: {params['sigma']}")
            
            # 필요한 경우 추가 필드 설정
            if 'rlwe' not in params:
                params['rlwe'] = 0
                print(f"   ✅ rlwe 추가: {params['rlwe']}")
            
            if 'min_hamming' not in params:
                params['min_hamming'] = params.get('hamming_weight', 3)
                print(f"   ✅ min_hamming 추가: {params['min_hamming']}")
                
            if 'max_hamming' not in params:
                params['max_hamming'] = params.get('hamming_weight', 3)
                print(f"   ✅ max_hamming 추가: {params['max_hamming']}")
            
            if 'secret_type' not in params:
                params['secret_type'] = 'binary'
                print(f"   ✅ secret_type 추가: {params['secret_type']}")
            
            if 'seed' not in params:
                params['seed'] = 111
                print(f"   ✅ seed 추가: {params['seed']}")
            
            if 'num_secret_seeds' not in params:
                params['num_secret_seeds'] = 1
                print(f"   ✅ num_secret_seeds 추가: {params['num_secret_seeds']}")
            
            if 'actions' not in params:
                params['actions'] = ['secrets']
                print(f"   ✅ actions 추가: {params['actions']}")
            
            if 'max_samples' not in params:
                params['max_samples'] = 2000000
                print(f"   ✅ max_samples 추가: {params['max_samples']}")
                
            if 'dump_path' not in params:
                params['dump_path'] = f'C:\\Users\\user\\OneDrive\\Desktop\\salsa-repro-salsa-connected\\data\\precomputed\\{dataset}\\'
                print(f"   ✅ dump_path 추가")
                
            if 'exp_name' not in params:
                params['exp_name'] = dataset
                print(f"   ✅ exp_name 추가: {params['exp_name']}")
            
            # 수정된 params 저장
            with open(params_path, 'wb') as f:
                pickle.dump(params, f)
            
            print(f"   ✅ {dataset} 수정 완료!\n")
            
        except Exception as e:
            print(f"   ❌ {dataset} 수정 실패: {e}\n")

if __name__ == "__main__":
    print("🔧 이진 데이터셋 params.pkl 파일들을 SALSA 호환성을 위해 수정합니다...")
    fix_binary_params()
    print("🎉 모든 수정 완료!")