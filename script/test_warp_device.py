import warp as wp
import torch

wp.init()

print(f"Torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Torch device count: {torch.cuda.device_count()}")
    print(f"Torch current device: {torch.cuda.current_device()}")

print(f"Warp devices: {wp.get_devices()}")

try:
    d = wp.get_device("cuda:0")
    print(f"Warp got device cuda:0: {d}")
except Exception as e:
    print(f"Warp failed to get cuda:0: {e}")

try:
    d = wp.get_device("cpu")
    print(f"Warp got device cpu: {d}")
except Exception as e:
    print(f"Warp failed to get cpu: {e}")
