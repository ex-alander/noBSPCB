import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.model import hybrid_predict_single

def main():
    model = YOLO("models/best.pt")
    result = hybrid_predict_single("data/sample.jpg", model)
    print(result)

if __name__ == "__main__":
    main()
