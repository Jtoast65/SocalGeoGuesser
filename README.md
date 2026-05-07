# SoCalGuessr Project

This workspace now includes:

- `train.py`: local training script for the Southern California city classifier.
- `predict.py`: Gradescope submission entry point.
- `model_utils.py`: shared model, dataset, transform, and checkpoint helpers.
- `report_template.md`: a starter outline for the written report.

## Training

Run a local train/validation experiment with:

```bash
python3 train.py --data-dir data --output-dir artifacts
```

Useful options:

```bash
python3 train.py --epochs 15 --batch-size 48 --freeze-epochs 2
python3 train.py --no-pretrained
python3 train.py --resume artifacts/model_weights.pt
```

The script saves:

- `artifacts/model_weights.pt`
- `artifacts/training_history.csv`
- `artifacts/training_curve.png`
- `artifacts/validation_confusion_matrix.csv`


## Notes

- The default model is `MobileNetV3-Small`, which keeps the checkpoint comfortably under the 50MB limit.
- The evaluation transform preserves a panoramic aspect ratio by resizing to `224x384` instead of forcing a square crop.
- `predict.py` runs entirely on CPU and only uses local weights, which is closer to the Gradescope environment.
