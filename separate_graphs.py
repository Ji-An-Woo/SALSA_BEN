import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_loss_accuracy_graphs():
    """Loss와 Accuracy를 별도 그래프로 생성"""
    
    # 샘플 데이터 (첨부된 이미지 기반)
    steps = np.arange(0, 350, 10)
    
    # Loss 데이터 (감소 추세)
    loss = 4.8 - 1.2 * np.log(steps + 1) + 0.3 * np.random.randn(len(steps))
    loss = np.maximum(loss, 3.5)  # 최소값 제한
    
    # Top-1 Accuracy 데이터 (증가 추세)
    acc1 = 100 * (1 - np.exp(-steps/80)) + 2 * np.random.randn(len(steps))
    acc1 = np.maximum(acc1, 0)  # 최소값 0
    acc1 = np.minimum(acc1, 12)  # 최대값 12
    
    # 1. Loss 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss, 'r-', linewidth=2, label='Training Loss')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('SALSA Training Progress: Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Loss 그래프 저장: graphs/loss_curve.png")
    
    # 2. Accuracy 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(steps, acc1, 'orange', linewidth=2, label='Top-1 Accuracy')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('SALSA Training Progress: Accuracy Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Accuracy 그래프 저장: graphs/accuracy_curve.png")

def create_learning_rate_graph():
    """Learning Rate Schedule 그래프 생성"""
    
    steps = np.arange(0, 300, 5)
    # Warmup + Linear decay 스케줄
    warmup_steps = 50
    lr = np.zeros_like(steps, dtype=float)
    
    for i, step in enumerate(steps):
        if step <= warmup_steps:
            lr[i] = 1e-7 + (5e-6 - 1e-7) * (step / warmup_steps)
        else:
            lr[i] = 5e-6 * (1 - (step - warmup_steps) / (300 - warmup_steps))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lr * 1e6, 'teal', linewidth=2)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Learning Rate (×10⁻⁶)', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/learning_rate_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Learning Rate 그래프 저장: graphs/learning_rate_curve.png")

def create_secret_recovery_graphs():
    """Secret Recovery 성능 그래프들 생성"""
    
    # 1. Epoch별 Recovery Success 바 그래프
    epochs = ['Epoch 1', 'Epoch 2']
    success_rates = [100, 100]  # 두 에포크 모두 성공
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(epochs, success_rates, color=['red', 'red'], alpha=0.8, width=0.6)
    plt.ylim(0, 120)
    plt.ylabel('Recovery Success (%)', fontsize=12)
    plt.title('Secret Recovery Success by Epoch', fontsize=14, fontweight='bold')
    
    # 바 위에 텍스트 추가
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('graphs/secret_recovery_success.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Secret Recovery Success 그래프 저장: graphs/secret_recovery_success.png")
    
    # 2. Recovery Accuracy over Epochs
    epochs_cont = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    top1_recovery = np.array([3, 15, 15.5, 15.2, 15.8, 16.0, 16.2])
    top5_recovery = np.array([28, 29, 29.2, 29.0, 29.5, 29.8, 30.0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_cont, top1_recovery, 'o-', color='lightblue', linewidth=2, 
             markersize=6, label='Recovery Top-1 Acc')
    plt.plot(epochs_cont, top5_recovery, 's-', color='purple', linewidth=2, 
             markersize=6, label='Recovery Top-5 Acc')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recovery Accuracy (%)', fontsize=12)
    plt.title('Secret Recovery Performance Over Training', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/recovery_accuracy_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Recovery Accuracy 그래프 저장: graphs/recovery_accuracy_over_time.png")

def create_summary_info_box():
    """SALSA 실행 정보 요약 박스 생성"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # 정보 텍스트
    info_text = """
    🎯 SALSA 이진 비밀키 실험 결과
    
    📊 데이터셋: baseline_n10_binary
    • N: 10 (문제 차원)
    • Q: 842,779 (모듈러스)
    • 비밀키 타입: 이진 {0, 1}
    • Hamming Weight: 3
    
    🔧 모델 설정:
    • Loss: 3.744
    • Top-1 Accuracy: 12.11%
    • Top-5 Accuracy: 26.27%
    
    📈 학습 결과:
    • 실제 비밀키: 2/10
    • 비밀키 복구: 성공!
    • Epochs: [1, 2]
    
    ✅ 결론: LWE 이진 비밀키에 대한 성공적인 학습!
    """
    
    # 배경 박스
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8)
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=bbox_props, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('graphs/salsa_summary_info.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ SALSA 요약 정보 저장: graphs/salsa_summary_info.png")

def main():
    """모든 그래프 생성"""
    
    # 그래프 폴더 생성
    Path('graphs').mkdir(exist_ok=True)
    
    print("🎨 SALSA 그래프 분리 생성 시작...")
    print()
    
    # 각 그래프 생성
    create_loss_accuracy_graphs()
    print()
    
    create_learning_rate_graph()
    print()
    
    create_secret_recovery_graphs()
    print()
    
    create_summary_info_box()
    print()
    
    print("🎉 모든 그래프 생성 완료!")
    print("📁 생성된 파일들:")
    print("   - graphs/loss_curve.png")
    print("   - graphs/accuracy_curve.png") 
    print("   - graphs/learning_rate_curve.png")
    print("   - graphs/secret_recovery_success.png")
    print("   - graphs/recovery_accuracy_over_time.png")
    print("   - graphs/salsa_summary_info.png")

if __name__ == "__main__":
    main()