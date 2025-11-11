import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 첫 번째 그래프: Loss & Accuracy 분리
def create_loss_accuracy_plots():
    """Training Loss와 Accuracy를 각각 별도 그래프로 생성"""
    
    # 훈련 데이터 (첨부된 이미지를 기반으로 재현)
    training_steps = np.linspace(0, 300, 50)
    
    # Loss 데이터 (감소하는 패턴)
    loss_data = 4.8 - 1.2 * np.log(training_steps + 1) + 0.3 * np.sin(training_steps * 0.1) * np.exp(-training_steps * 0.01)
    
    # Accuracy 데이터 (증가하는 패턴)
    acc_data = 12 * (1 - np.exp(-training_steps * 0.015)) + np.random.normal(0, 0.3, len(training_steps))
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Loss 그래프
    ax1.plot(training_steps, loss_data, color='#FF6B6B', linewidth=2.5, label='Training Loss', marker='o', markersize=3)
    ax1.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#FF6B6B')
    ax1.set_title('🔥 Training Loss Progress', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Loss 축 범위 설정
    ax1.set_ylim(3.5, 5.0)
    ax1.tick_params(axis='y', labelcolor='#FF6B6B')
    
    # 2. Accuracy 그래프  
    ax2.plot(training_steps, acc_data, color='#4ECDC4', linewidth=2.5, label='Top-1 Accuracy (%)', marker='s', markersize=3)
    ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#4ECDC4')
    ax2.set_title('📈 Training Accuracy Progress', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    # Accuracy 축 범위 설정
    ax2.set_ylim(0, 15)
    ax2.tick_params(axis='y', labelcolor='#4ECDC4')
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.savefig('training_loss_accuracy_separated.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Loss & Accuracy 분리 그래프 저장: training_loss_accuracy_separated.png")

def create_learning_rate_plot():
    """Learning Rate Schedule 별도 그래프 생성"""
    
    # Learning Rate 데이터
    training_steps = np.linspace(0, 300, 50)
    lr_data = np.linspace(1.0, 5.0, len(training_steps))  # 선형 증가 패턴
    
    plt.figure(figsize=(10, 6))
    plt.plot(training_steps, lr_data, color='#45B7D1', linewidth=3, marker='o', markersize=4)
    plt.xlabel('Training Step', fontsize=12, fontweight='bold')
    plt.ylabel('Learning Rate', fontsize=12, fontweight='bold')
    plt.title('📊 Learning Rate Schedule', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')
    
    # 배경 스타일
    plt.tight_layout()
    plt.savefig('learning_rate_schedule.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Learning Rate Schedule 그래프 저장: learning_rate_schedule.png")

def create_secret_recovery_plots():
    """Secret Recovery 성능 그래프들을 분리하여 생성"""
    
    # 데이터 설정
    epochs = [0, 1, 2, 3]
    
    # Recovery Success 데이터
    success_data = [0, 1, 1, 1]  # 에포크 1, 2에서 성공
    
    # Recovery Accuracy 데이터
    top1_acc = [15, 15.5, 15.2, 16.1]
    top5_acc = [28, 28.5, 28.2, 29.1]
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Secret Recovery Success 바 차트
    colors = ['#FF6B6B' if x == 0 else '#4ECDC4' for x in success_data]
    bars = ax1.bar([f'Epoch {i}' for i in epochs], success_data, color=colors, alpha=0.8, width=0.6)
    
    ax1.set_ylabel('Recovery Success', fontsize=12, fontweight='bold')
    ax1.set_title('🎯 Secret Recovery Success', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(-0.1, 1.5)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#f8f9fa')
    
    # 성공/실패 라벨 추가
    for i, (bar, success) in enumerate(zip(bars, success_data)):
        label = 'Success' if success == 1 else 'Failed'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                label, ha='center', va='bottom', fontweight='bold')
    
    # 2. Recovery Accuracy 선 그래프
    ax2.plot(epochs, top1_acc, color='#FF9F43', marker='o', linewidth=3, markersize=8, 
            label='Recovery Top-1 Acc', alpha=0.9)
    ax2.plot(epochs, top5_acc, color='#6C5CE7', marker='s', linewidth=3, markersize=8, 
            label='Recovery Top-5 Acc', alpha=0.9)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recovery Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('📊 Secret Recovery Performance', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.set_facecolor('#f8f9fa')
    ax2.set_ylim(10, 35)
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.savefig('secret_recovery_performance.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Secret Recovery 성능 그래프 저장: secret_recovery_performance.png")

if __name__ == "__main__":
    print("🎨 첨부된 그래프들을 각각 분리하여 생성합니다...")
    print()
    
    # 1. Loss & Accuracy 분리
    create_loss_accuracy_plots()
    print()
    
    # 2. Learning Rate Schedule 
    create_learning_rate_plot()
    print()
    
    # 3. Secret Recovery 성능
    create_secret_recovery_plots()
    print()
    
    print("🎉 모든 그래프 분리 완료!")
    print("📁 생성된 파일들:")
    print("   - training_loss_accuracy_separated.png")
    print("   - learning_rate_schedule.png") 
    print("   - secret_recovery_performance.png")