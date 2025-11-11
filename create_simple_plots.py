import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 생성
import matplotlib.pyplot as plt
import numpy as np

def create_simple_plots():
    """첨부된 그래프들을 각각 분리하여 간단하게 생성"""
    
    print("🎨 그래프 생성 시작...")
    
    # 1. Training Loss & Accuracy 분리
    training_steps = np.linspace(0, 300, 50)
    
    # Loss 그래프
    plt.figure(figsize=(10, 6))
    loss_data = 4.8 - 1.2 * np.log(training_steps + 1) + 0.3 * np.sin(training_steps * 0.1) * np.exp(-training_steps * 0.01)
    plt.plot(training_steps, loss_data, color='red', linewidth=2, label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Training Loss 그래프 저장: training_loss.png")
    
    # Accuracy 그래프
    plt.figure(figsize=(10, 6))
    acc_data = 12 * (1 - np.exp(-training_steps * 0.015)) + np.random.normal(0, 0.3, len(training_steps))
    plt.plot(training_steps, acc_data, color='orange', linewidth=2, label='Top-1 Accuracy (%)')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Training Accuracy 그래프 저장: training_accuracy.png")
    
    # 2. Learning Rate Schedule
    plt.figure(figsize=(10, 6))
    lr_data = np.linspace(1.0, 5.0, len(training_steps))
    plt.plot(training_steps, lr_data, color='cyan', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_rate_schedule.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Learning Rate Schedule 그래프 저장: learning_rate_schedule.png")
    
    # 3. Secret Recovery Success
    epochs = [0, 1, 2, 3]
    success_data = [0, 1, 1, 1]
    
    plt.figure(figsize=(8, 6))
    colors = ['red' if x == 0 else 'green' for x in success_data]
    plt.bar([f'Epoch {i}' for i in epochs], success_data, color=colors, alpha=0.7)
    plt.ylabel('Recovery Success')
    plt.title('Secret Recovery Success by Epoch')
    plt.ylim(0, 1.2)
    plt.savefig('secret_recovery_success.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Secret Recovery Success 그래프 저장: secret_recovery_success.png")
    
    # 4. Secret Recovery Accuracy
    top1_acc = [15, 15.5, 15.2, 16.1]
    top5_acc = [28, 28.5, 28.2, 29.1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, top1_acc, color='blue', marker='o', linewidth=2, label='Recovery Top-1 Acc')
    plt.plot(epochs, top5_acc, color='green', marker='s', linewidth=2, label='Recovery Top-5 Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Recovery Accuracy (%)')
    plt.title('Secret Recovery Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('secret_recovery_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Secret Recovery Accuracy 그래프 저장: secret_recovery_accuracy.png")
    
    print("\n🎉 모든 그래프 생성 완료!")
    print("📁 생성된 파일들:")
    print("   - training_loss.png")
    print("   - training_accuracy.png")
    print("   - learning_rate_schedule.png")
    print("   - secret_recovery_success.png")
    print("   - secret_recovery_accuracy.png")

if __name__ == "__main__":
    create_simple_plots()