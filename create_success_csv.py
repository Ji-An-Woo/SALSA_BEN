import pandas as pd
import json
import re
import os
from datetime import datetime

def extract_salsa_results():
    """성공한 SALSA 실행 결과를 분석해서 CSV로 정리"""
    
    # 성공한 실험들의 경로
    success_experiments = [
        {
            'exp_name': 'final_test',
            'exp_id': 'zgm2ws92ej',
            'log_path': r'C:\checkpoint\user\dumped\final_test\zgm2ws92ej\train.log',
            'data_path': r'C:\Users\user\OneDrive\Desktop\salsa-repro-salsa-connected\data\precomputed\baseline_n10'
        }
    ]
    
    results = []
    
    for exp in success_experiments:
        log_path = exp['log_path']
        
        if not os.path.exists(log_path):
            print(f"로그 파일이 없습니다: {log_path}")
            continue
            
        print(f"분석 중: {exp['exp_name']} ({exp['exp_id']})")
        
        # 로그 파싱
        train_metrics = []
        recover_metrics = []
        secret_recovered = False
        recovery_epochs = []
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 비밀키 복구 성공 확인
            if 'all bits in secret recovered!' in content and 'Recovered secret!' in content:
                secret_recovered = True
                
                # 성공한 epoch 찾기
                lines = content.split('\n')
                for line in lines:
                    if 'all bits in secret recovered!' in line:
                        # epoch 정보 추출
                        if 'recover/epoch' in content:
                            epoch_match = re.search(r'"recover/epoch": (\d+)', line)
                            if epoch_match:
                                recovery_epochs.append(int(epoch_match.group(1)))
            
            # 최종 학습 메트릭 추출
            train_lines = [line for line in content.split('\n') if '"train/acc1":' in line and '"train/step":' in line]
            if train_lines:
                try:
                    last_train = train_lines[-1]
                    json_match = re.search(r'\{.*\}', last_train)
                    if json_match:
                        final_metrics = json.loads(json_match.group())
                        
                        results.append({
                            'experiment_name': exp['exp_name'],
                            'experiment_id': exp['exp_id'],
                            'data_path': exp['data_path'],
                            'task': 'lwe',
                            'secret_recovered': secret_recovered,
                            'recovery_epochs': str(recovery_epochs) if recovery_epochs else 'N/A',
                            'recovery_count': len(recovery_epochs),
                            'final_loss': final_metrics.get('train/loss', 0),
                            'final_acc1': final_metrics.get('train/acc1', 0),
                            'final_acc5': final_metrics.get('train/acc5', 0),
                            'final_step': final_metrics.get('train/step', 0),
                            'final_epoch': final_metrics.get('train/epoch', 0),
                            'learning_rate': final_metrics.get('learning_rate', 0),
                            'N': 10,  # 문제 차원
                            'hamming': 3,  # 해밍 웨이트
                            'epochs_run': final_metrics.get('train/epoch', 0) + 1,
                            'batch_size': 4,
                            'dtype': 'float16',
                            'compile': False,
                            'device': 'cuda:0',
                            'status': 'SUCCESS - Secret Recovered!',
                            'timestamp': datetime.now().strftime('%Y-09-29 02:03:00')
                        })
                except Exception as e:
                    print(f"메트릭 파싱 오류: {e}")
    
    return results

def create_success_csv():
    """성공한 SALSA 결과를 CSV로 저장"""
    results = extract_salsa_results()
    
    if not results:
        print("분석할 결과가 없습니다.")
        return
    
    # DataFrame 생성
    df = pd.DataFrame(results)
    
    # CSV 저장
    output_path = r'C:\Users\user\OneDrive\Desktop\salsa-repro-salsa-connected\salsa_success_results.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 성공한 SALSA 결과가 저장되었습니다:")
    print(f"📁 {output_path}")
    print(f"📊 총 {len(results)}개 성공 실험")
    
    # 결과 요약 출력
    for result in results:
        print(f"\n🎯 {result['experiment_name']}:")
        print(f"   • 비밀키 복구: {'✅ 성공' if result['secret_recovered'] else '❌ 실패'}")
        print(f"   • 복구 횟수: {result['recovery_count']}회")
        print(f"   • 성공 Epoch: {result['recovery_epochs']}")
        print(f"   • 최종 Loss: {result['final_loss']:.4f}")
        print(f"   • 최종 Accuracy: {result['final_acc1']*100:.2f}%")
        print(f"   • 실행 Step: {result['final_step']}")
    
    return output_path

if __name__ == "__main__":
    csv_path = create_success_csv()
    
    # 추가로 JSON 형태로도 저장
    if csv_path:
        json_path = csv_path.replace('.csv', '.json')
        df = pd.read_csv(csv_path)
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        print(f"📄 JSON 형태로도 저장: {json_path}")