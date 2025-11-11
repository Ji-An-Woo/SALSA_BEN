#!/usr/bin/env python3
"""
GPU Setup Helper for SALSA-repro
=================================

This script helps you set up CUDA/GPU support for running SALSA experiments.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the result"""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success!")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"❌ Error!")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_nvidia_driver():
    """Check if NVIDIA driver is installed"""
    print("\n🔍 Checking NVIDIA driver...")
    if run_command("nvidia-smi", "Checking NVIDIA driver"):
        return True
    else:
        print("⚠️ NVIDIA driver not found. Please install NVIDIA drivers first.")
        print("Visit: https://www.nvidia.com/drivers/")
        return False

def check_conda():
    """Check if conda is available"""
    print("\n🔍 Checking conda installation...")
    return run_command("conda --version", "Checking conda")

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("\n🚀 Installing PyTorch with CUDA support...")
    
    # Try to detect CUDA version
    cuda_version = "11.8"  # Default fallback
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode == 0:
            # Simple CUDA version detection (you might need to adjust this)
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    break
    except:
        pass
    
    print(f"Detected/Using CUDA version: {cuda_version}")
    
    # Install PyTorch with CUDA
    if cuda_version.startswith("12"):
        pip_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif cuda_version.startswith("11"):
        pip_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        pip_cmd = "pip install torch torchvision torchaudio"
    
    return run_command(pip_cmd, "Installing PyTorch with CUDA")

def test_pytorch_cuda():
    """Test if PyTorch can use CUDA"""
    print("\n🧪 Testing PyTorch CUDA support...")
    test_script = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
'''
    
    with open('test_cuda.py', 'w') as f:
        f.write(test_script)
    
    success = run_command("python test_cuda.py", "Testing CUDA support")
    os.remove('test_cuda.py')
    return success

def main():
    print("🎯 SALSA-repro GPU Setup Helper")
    print("=" * 50)
    
    # Check NVIDIA driver
    if not check_nvidia_driver():
        print("\n❌ Setup cannot continue without NVIDIA drivers.")
        return False
    
    # Install PyTorch with CUDA
    if not install_pytorch_cuda():
        print("\n❌ Failed to install PyTorch with CUDA.")
        return False
    
    # Install other requirements
    if not run_command("pip install -r requirements.txt", "Installing other requirements"):
        print("\n⚠️ Some requirements may not have installed correctly.")
    
    # Test CUDA support
    if test_pytorch_cuda():
        print("\n🎉 GPU setup completed successfully!")
        print("You can now run the SALSA experiments with GPU acceleration.")
        return True
    else:
        print("\n❌ GPU setup verification failed.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)