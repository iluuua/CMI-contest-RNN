# CMI-contest-RNN
A compact reproducible training + inference pipeline for the CMI sensor-data contest. Implements an RNN + attention classifier, Hydra-based config, training script, inference script and a Kaggle-ready notebook.

## Repo Layout

├─ conf/                       # hydra YAML configs (data, model, trainer)
│  ├─ config.yaml
│  └─ data.yaml
├─ train.py                    # Hydra entrypoint for training
├─ predict.py                  # CLI for batch inference (writes CSV)
├─ notebook-solution.ipynb     # Kaggle notebook (train/test or inference-server)
├─ src/                        # project package (model, data, utils)
│  ├─ model.py
│  ├─ data.py
│  └─ utils.py
├─ requirements.txt
├─ README.md
└─ outputs/                    # runtime outputs (Hydra run dirs)

## Quickstart
1. `pip install -r requirements.txt`
2. `python train.py trainer.batch_size=64 trainer.epochs=30`
3. `python predict.py --artifacts outputs/<run> --csv ./test.csv --out submission.csv`
