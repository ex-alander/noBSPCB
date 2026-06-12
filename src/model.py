import numpy as np
from ultralytics import YOLO
from src.mcd import mcd_predict_single

# Экспериментально подобранный порог неопределённости
UNCERTAINTY_THRESHOLD = 0.02

def hybrid_predict_single(image_path, model, num_passes=30, conf_thres=0.25):
    """
    Гибридный инференс:
    - baseline (один прогон)
    - если дефектов нет -> статус 'passed'
    - если есть дефекты -> MCD, решение по variance
    """
    results = model(image_path, conf=conf_thres, verbose=False)
    detections = results[0]

    if detections.boxes is None or len(detections.boxes) == 0:
        return {
            'status': 'passed',
            'details': {'num_defects_found': 0, 'uncertainty': None, 'mcd_performed': False}
        }

    clusters = mcd_predict_single(model, image_path, num_passes=num_passes)
    if not clusters:
        return {
            'status': 'uncertain',
            'details': {
                'num_defects_found': len(detections.boxes),
                'uncertainty': 1.0,
                'mcd_performed': True
            }
        }

    avg_variance = np.mean([c['variance'] for c in clusters])
    status = 'defect' if avg_variance < UNCERTAINTY_THRESHOLD else 'uncertain'

    return {
        'status': status,
        'details': {
            'num_defects_found': len(detections.boxes),
            'uncertainty': avg_variance,
            'clusters': clusters,
            'mcd_performed': True
        }
    }
