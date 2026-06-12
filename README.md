# SureDefect — PCB Defect Detection with Uncertainty Estimation

**SureDefect** — hybrid system for PCB defect detection (missing hole, mouse bite, open circuit, short, spur, spurious copper) based on YOLOv8n and Monte Carlo Dropout.

**Key result:** False positives reduced by **91.5%** (from 352 to 30) while maintaining Recall = 0.987.

## Comparison with baseline

| Metric | Baseline YOLOv8n | SureDefect (Hybrid MCD) |
|--------|------------------|--------------------------|
| mAP@0.5 | 0.988 | 0.978 (-1.0%) |
| False Positives | 352 | 30 (-91.5%) |
| Recall | 0.987 | 0.987 |
| Time per image (CPU) | 31 ms | 43 ms* |

*\* Weighted average assuming 2% defective boards. MCD runs only when a defect is detected.*

## Quick start

```bash
git clone https://github.com/your-username/suredefect.git
cd suredefect
pip install -r requirements.txt
jupyter lab notebooks/
```

## License

MIT

## CI Status

[![CI](https://github.com/your-username/suredefect/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/suredefect/actions/workflows/ci.yml)

