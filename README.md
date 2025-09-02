## CMI-contest-RNN
A compact reproducible training + inference pipeline for the CMI sensor-data contest. Implements an RNN + attention classifier, Hydra-based config, training script, inference script and a Kaggle-ready notebook.

### Repo Layout
```
.
├─ conf/                                       # hydra config groups
│ ├─ config.yaml
│ ├─ data.yaml
│ ├─ model.yaml
│ └─ trainer.yaml
├─ train.py                                    # Hydra entrypoint: trains model, saves artifacts
├─ predict.py                                  # Batch inference CLI / helper for local prediction
├─ notebook-solution.ipynb                     # Kaggle-ready notebook 
├─ requirements.txt
├─ outputs/                                    # Hydra run dirs and artifacts 
```
### Quickstart
1. `pip install -r requirements.txt`
2. `python train.py trainer.batch_size=64 trainer.epochs=30`
3. `python predict.py --artifacts outputs/<run> --csv ./test.csv --out submission.csv`
