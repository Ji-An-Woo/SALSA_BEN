import matplotlib.pyplot as plt
import json
import re
import numpy as np
from datetime import datetime
import seaborn as sns

# 로그 파일 경로
log_path = r"C:\checkpoint\user\dumped\final_test\zgm2ws92ej\train.log"

def parse_log_file(log_path):
    """로그 파일을 파싱해서 학습 메트릭과 복구 결과를 추출"""
    train_metrics = []
    recover_metrics = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 학습 메트릭 파싱
            if '"train/acc1":' in line and '"train/step":' in line:
                try:
                    # JSON 부분 추출
                    json_match = re.search(r'\{.*\}', line)
                    if json_match:
                        data = json.loads(json_match.group())
                        train_metrics.append({
                            'step': data.get('train/step', 0),
                            'epoch': data.get('train/epoch', 0),
                            'loss': data.get('train/loss', 0),
                            'acc1': data.get('train/acc1', 0),
                            'acc5': data.get('train/acc5', 0),
                            'learning_rate': data.get('learning_rate', 0)
                        })
                except:
                    continue
            
            # 복구 메트릭 파싱
            elif '"recover/matched":' in line:
                try:
                    json_match = re.search(r'\{.*\}', line)
                    if json_match:
                        data = json.loads(json_match.group())
                        recover_metrics.append({
                            'epoch': data.get('recover/epoch', 0),
                            'acc1': data.get('recover/acc1', 0),
                            'acc5': data.get('recover/acc5', 0),
                            'loss': data.get('recover/loss', 0),
                            'matched': data.get('recover/matched', False)
                        })
                except:
                    continue
    
    return train_metrics, recover_metrics

def plot_salsa_results():
    """SALSA 결과를 시각화"""
    print("로그 파일 파싱 중...")
    train_metrics, recover_metrics = parse_log_file(log_path)
    
    print(f"학습 메트릭: {len(train_metrics)}개")
    print(f"복구 메트릭: {len(recover_metrics)}개")
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # 색상 팔레트
    colors = sns.color_palette("husl", 8)
    
    # 1. 학습 Loss 및 Accuracy
    ax1 = plt.subplot(2, 3, 1)
    if train_metrics:
        steps = [m['step'] for m in train_metrics]
        losses = [m['loss'] for m in train_metrics]
        acc1 = [m['acc1'] * 100 for m in train_metrics]  # 백분율로 변환
        
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(steps, losses, color=colors[0], linewidth=2, marker='o', markersize=4, label='Loss')
        line2 = ax1_twin.plot(steps, acc1, color=colors[1], linewidth=2, marker='s', markersize=4, label='Top-1 Accuracy (%)')
        
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss', color=colors[0])
        ax1_twin.set_ylabel('Accuracy (%)', color=colors[1])
        ax1.set_title('🎯 Training Progress: Loss & Accuracy', fontsize=14, fontweight='bold')
        
        # 범례
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        ax1.grid(True, alpha=0.3)
    
    # 2. Top-1 vs Top-5 Accuracy
    ax2 = plt.subplot(2, 3, 2)
    if train_metrics:
        acc1 = [m['acc1'] * 100 for m in train_metrics]
        acc5 = [m['acc5'] * 100 for m in train_metrics]
        
        ax2.plot(steps, acc1, color=colors[2], linewidth=2, marker='o', markersize=4, label='Top-1 Accuracy')
        ax2.plot(steps, acc5, color=colors[3], linewidth=2, marker='^', markersize=4, label='Top-5 Accuracy')
        
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('📈 Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate
    ax3 = plt.subplot(2, 3, 3)
    if train_metrics:
        lr = [m['learning_rate'] for m in train_metrics]
        
        ax3.plot(steps, lr, color=colors[4], linewidth=2, marker='d', markersize=4)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('📊 Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax3.grid(True, alpha=0.3)
    
    # 4. 비밀키 복구 성공 시점
    ax4 = plt.subplot(2, 3, 4)
    if recover_metrics:
        epochs = [m['epoch'] for m in recover_metrics]
        matched = [1 if m['matched'] else 0 for m in recover_metrics]
        colors_recover = ['red' if m else 'gray' for m in matched]
        
        bars = ax4.bar(epochs, matched, color=colors_recover, alpha=0.7, edgecolor='black', linewidth=1)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recovery Success')
        ax4.set_title('🔐 Secret Recovery Success', fontsize=14, fontweight='bold')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Failed', 'Success'])
        
        # 성공한 epoch 표시
        success_epochs = [epochs[i] for i, m in enumerate(matched) if m == 1]
        for epoch in success_epochs:
            ax4.text(epoch, 1.1, f'✅ Epoch {epoch}', ha='center', va='bottom', 
                    fontweight='bold', color='red', fontsize=10)
        
        ax4.grid(True, alpha=0.3)
    
    # 5. 복구 성능 메트릭
    ax5 = plt.subplot(2, 3, 5)
    if recover_metrics:
        epochs = [m['epoch'] for m in recover_metrics]
        recover_acc1 = [m['acc1'] * 100 for m in recover_metrics]
        recover_acc5 = [m['acc5'] * 100 for m in recover_metrics]
        
        ax5.plot(epochs, recover_acc1, color=colors[5], linewidth=3, marker='o', markersize=8, label='Recovery Top-1 Acc')
        ax5.plot(epochs, recover_acc5, color=colors[6], linewidth=3, marker='s', markersize=8, label='Recovery Top-5 Acc')
        
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Recovery Accuracy (%)')
        ax5.set_title('🎯 Secret Recovery Performance', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. 종합 결과 요약
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 결과 요약 텍스트
    if train_metrics and recover_metrics:
        final_loss = train_metrics[-1]['loss']
        final_acc1 = train_metrics[-1]['acc1'] * 100
        final_acc5 = train_metrics[-1]['acc5'] * 100
        
        success_count = sum(1 for m in recover_metrics if m['matched'])
        total_attempts = len(recover_metrics)
        
        summary_text = f"""
🎉 SALSA 실행 결과 요약

📊 최종 학습 성능:
   • Loss: {final_loss:.3f}
   • Top-1 Accuracy: {final_acc1:.2f}%
   • Top-5 Accuracy: {final_acc5:.2f}%

🔐 비밀키 복구 성공:
   • 성공 횟수: {success_count}/{total_attempts}
   • 성공률: {(success_count/total_attempts*100):.1f}%

✅ 복구 성공 Epoch: {[m['epoch'] for m in recover_metrics if m['matched']]}

🚀 결론: LWE 암호 문제 해결 성공!
        """
        
        ax6.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                family='monospace')
    
    plt.tight_layout(pad=3.0)
    plt.suptitle('🔥 SALSA: Learning With Errors 암호 해독 성공! 🔥', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 저장 및 표시
    output_path = r"C:\Users\user\OneDrive\Desktop\salsa-repro-salsa-connected\salsa_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n플롯이 저장되었습니다: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    try:
        plot_salsa_results()
        print("\n🎉 SALSA 결과 시각화 완료!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()