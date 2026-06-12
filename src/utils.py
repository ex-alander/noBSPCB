import cv2
import torch
import numpy as np
from pathlib import Path

def load_image(path: Path, target_size: int = 640):
    """Загружает изображение, ресайзит и нормализует."""
    img = cv2.imread(str(path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    input_tensor = torch.from_numpy(img_resized).float().permute(2,0,1).unsqueeze(0) / 255.0
    return img_resized, input_tensor

def setup_device():
    """Возвращает устройство (cpu/mps/cuda)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int = 42):
    """Фиксирует случайные сиды для воспроизводимости."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
