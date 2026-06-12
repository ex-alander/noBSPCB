import cv2
import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression

from src.utils import load_image, setup_device


def enable_dropout(model_module):
    """Включает Dropout слои для инференса."""
    for m in model_module.modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd):
            m.train()


def iou(box1, box2):
    """IoU для двух bounding boxes [x1,y1,x2,y2]."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def cluster_boxes(all_boxes, iou_threshold=0.5):
    """Кластеризует боксы из разных прогонов MCD."""
    if not all_boxes:
        return []
    flat_boxes = [{"box": b, "used": False} for b in all_boxes]
    clusters = []
    for i in range(len(flat_boxes)):
        if flat_boxes[i]["used"]:
            continue
        cluster = [i]
        flat_boxes[i]["used"] = True
        for j in range(i + 1, len(flat_boxes)):
            if flat_boxes[j]["used"]:
                continue
            if iou(flat_boxes[i]["box"][:4], flat_boxes[j]["box"][:4]) >= iou_threshold:
                cluster.append(j)
                flat_boxes[j]["used"] = True
        clusters.append(cluster)
    return clusters


def mcd_predict_single(
    model,
    image_path,
    num_passes=30,
    raw_conf=0.15,
    final_conf=0.25,
    iou_nms=0.45,
    iou_cluster=0.5,
):
    """MC Dropout инференс для одного изображения."""
    device = setup_device()
    img_resized, input_tensor = load_image(image_path)
    input_tensor = input_tensor.to(device)

    original_state = model.model.training
    model.model.eval()
    enable_dropout(model.model)

    all_boxes = []
    for _ in range(num_passes):
        with torch.no_grad():
            preds = model.model(input_tensor)
        outputs = non_max_suppression(preds, conf_thres=raw_conf, iou_thres=iou_nms)
        if outputs[0] is not None and len(outputs[0]):
            for det in outputs[0].cpu().numpy():
                all_boxes.append([float(x) for x in det[:6]])

    model.model.train(original_state)

    all_detections = [b for b in all_boxes if b[4] >= final_conf]
    if not all_detections:
        return []

    clusters_idx = cluster_boxes(all_detections, iou_threshold=iou_cluster)
    cluster_metrics = []
    for cluster in clusters_idx:
        confs = [all_detections[idx][4] for idx in cluster]
        classes = [int(all_detections[idx][5]) for idx in cluster]
        class_counts = np.bincount(classes, minlength=6)
        class_probs = class_counts / len(classes)
        entropy = -np.sum(class_probs * np.log(class_probs + 1e-8))
        variance = np.var(confs) if len(confs) > 1 else 0.0
        avg_box = np.mean([all_detections[idx][:4] for idx in cluster], axis=0)
        cluster_metrics.append(
            {
                "bbox": avg_box.tolist(),
                "class": int(np.argmax(class_counts)),
                "confidence": np.mean(confs),
                "entropy": entropy,
                "variance": variance,
                "num_passes": len(cluster),
            }
        )
    return cluster_metrics
