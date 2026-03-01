import torch

print("--- Intel Arc GPU Check ---")
print(f"PyTorch version installed: {torch.__version__}")
print(f"Is XPU (Intel GPU) available?: {torch.xpu.is_available()}")

if hasattr(torch, "xpu") and torch.xpu.is_available():
    print(f"GPU Detected: {torch.xpu.get_device_name(0)}")
else:
    print("No Intel GPU detected. PyTorch will fall back to CPU.")