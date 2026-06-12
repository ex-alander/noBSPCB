import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.utils import load_image


def draw_detections(image_path, model, conf_thres=0.25):
    """Рисует рамки детекций на изображении."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(image_path, conf=conf_thres, verbose=False)[0]
    img_out = img_rgb.copy()
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            detections.append((xyxy, conf, cls))
            cv2.rectangle(
                img_out, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2
            )
            label = f"cls:{cls} conf:{conf:.2f}"
            cv2.putText(
                img_out,
                label,
                (xyxy[0], xyxy[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    return img_out, detections


def generate_eigencam(model, image_path, target_layers):
    """Генерирует тепловую карту EigenCAM."""
    img_resized, input_tensor = load_image(image_path)
    input_tensor = input_tensor.to(next(model.model.parameters()).device)
    with EigenCAM(model=model.model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        visualization = show_cam_on_image(
            img_resized / 255.0, grayscale_cam, use_rgb=True
        )
    return visualization
