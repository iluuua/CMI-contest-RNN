## CMI-contest-RNN

Compact, reproducible training + inference pipeline for the CMI "Detect Behavior with Sensor Data" challenge.  
The project implements a multivariate time-series classifier (RNN + attention pooling) that converts wrist/IMU sensor sequences into gesture labels and provides a Kaggle-ready inference server for submission.

(Problem context: sensor/time-series classification using the competition data.) :contentReference[oaicite:0]{index=0}

## What this repository does
- Preprocesses per-sequence multivariate sensor data (scaling, padding/truncation).
- Trains an RNN classifier (LSTM/GRU) with an attention pooling head to aggregate temporal features into a prediction.
- Exports deterministic artifacts for inference: model state, scaler, feature list, label mapping.
- Provides a Kaggle-compatible notebook that implements the `predict(sequence, demographics)` inference function and runs the official evaluation gateway locally for validation.  
RNN + attention is a practical and proven choice for activity / time-series classification tasks.

## Repo layout
```
.
├─ conf/                                       # runtime config files (data paths, model & trainer params)
│ ├─ config.yaml
│ ├─ data.yaml
│ ├─ model.yaml
│ └─ trainer.yaml
├─ train.py                                    # training entrypoint 
├─ predict.py                                  # batch inference CLI (loads artifacts → CSV output)
├─ notebook-solution.ipynb                     # Kaggle notebook (inference server + local gateway)
├─ requirements.txt
├─ outputs/                                    # training run directories and saved artifacts
```
## Quickstart
1. `pip install -r requirements.txt`
2. `python train.py trainer.batch_size=64 trainer.epochs=30`
3. `python predict.py --artifacts outputs/<run> --csv ./test.csv --out submission.csv`
