import os, json, argparse, glob, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def pad_to_len(X: np.ndarray, pad_len: int, pad_mode="post", trunc_mode="post", value: float = 0.0):
    X = np.asarray(X, dtype=np.float32)
    T, C = X.shape
    if T == pad_len:
        return X.copy()
    if T > pad_len:
        if trunc_mode == "post":
            out = X[:pad_len]
        elif trunc_mode == "pre":
            out = X[-pad_len:]
        elif trunc_mode == "center":
            start = (T - pad_len) // 2
            out = X[start:start + pad_len]
        else:
            raise ValueError("bad trunc_mode")
        return out.astype(np.float32, copy=False)
    pad_total = pad_len - T
    if pad_mode == "post":
        pad_before, pad_after = 0, pad_total
    elif pad_mode == "pre":
        pad_before, pad_after = pad_total, 0
    elif pad_mode == "center":
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
    else:
        raise ValueError("bad pad_mode")
    return np.pad(X, ((pad_before, pad_after), (0, 0)), mode="constant", constant_values=value).astype(np.float32, copy=False)

def load_config(art_dir: str):
    import yaml
    cfg_path = os.path.join(art_dir, "config_composed.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_artifacts(art_dir: str):
    with open(os.path.join(art_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(art_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    import joblib
    scaler = joblib.load(os.path.join(art_dir, "scaler.pkl"))
    state_path = os.path.join(art_dir, "model_state_dict.pt")
    cfg = load_config(art_dir)
    return meta, id2label, scaler, state_path, cfg

def find_latest_outputs(root="outputs"):
    cands = sorted(glob.glob(os.path.join(root, "*", "*")), key=os.path.getmtime, reverse=True)
    for d in cands:
        if os.path.isfile(os.path.join(d, "model_state_dict.pt")):
            return d
    raise FileNotFoundError("No artifacts found under outputs/*/*")

def load_test_sequences(csv_path: str, id_col: str, feat_cols: list[str]):
    df = pd.read_csv(csv_path)
    miss = [c for c in [id_col] + feat_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in test CSV: {miss}")
    X, ids = [], []
    for sid, g in df.groupby(id_col, sort=False):
        arr = g[feat_cols].to_numpy(dtype=np.float32)
        X.append(arr)
        ids.append(sid)
    return X, ids

class AttnPool(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, mask=None):
        A = torch.tanh(self.proj(H))
        e = self.v(A).squeeze(-1)             # (B,T)
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))
        w = torch.softmax(e, dim=1).unsqueeze(-1)
        ctx = (H * w).sum(dim=1)
        return self.drop(ctx), w.squeeze(-1)

class RNNWithAttn(nn.Module):
    def __init__(self, input_dim, num_classes, rnn_type="lstm",
                 hidden_size=128, num_layers=2, bidirectional=True,
                 input_dropout=0.1, attn_dropout=0.1, fc_dropout=0.2):
        super().__init__()

        self.in_drop = nn.Dropout(input_dropout)
        rnn = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn(input_dim, hidden_size, num_layers,
                       batch_first=True, bidirectional=bidirectional,
                       dropout=0.0 if num_layers == 1 else 0.1)
        d = hidden_size * (2 if bidirectional else 1)
        self.attn = AttnPool(d, dropout=attn_dropout)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(fc_dropout),
                                  nn.Linear(d, num_classes))

    def forward(self, x_pad, lengths):
        x = self.in_drop(x_pad)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        H, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x_pad.size(1))
        idx = torch.arange(x_pad.size(1), device=x_pad.device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)       # (B,T) bool
        ctx, _ = self.attn(H, mask=mask)
        return self.head(ctx)                    # (B,C)

# -------- dataset for inference --------
class SeqTestDataset(Dataset):
    def __init__(self, X_list, pad_len, scaler, pad_mode="post", trunc_mode="post"):
        self.X = X_list
        self.pad_len = int(pad_len)
        self.scaler = scaler
        self.pad_mode = pad_mode
        self.trunc_mode = trunc_mode
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.scaler.transform(self.X[idx])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        L = min(x.shape[0], self.pad_len)
        x = pad_to_len(x, self.pad_len, self.pad_mode, self.trunc_mode)
        return torch.from_numpy(x), int(L), idx

def collate_pred(batch):
    xs, lens, ids = zip(*batch)
    xs = torch.stack(xs, 0)
    lens = torch.tensor(lens).long()
    return xs, lens, ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, default="conf/data/dataset/test.csv")
    ap.add_argument("--artifacts", required=False, default=None, help="outputs/<date>/<time>")
    ap.add_argument("--out", required=False, default="submission.csv")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    art_dir = args.artifacts or find_latest_outputs("outputs")
    meta, id2label, scaler, state_path, cfg = load_artifacts(art_dir)

    id_col   = cfg["data"]["id_col"]
    label_col= cfg["data"]["label_col"]
    pad_len  = int(meta["pad_len"])
    feat_cols= meta["feat_cols"]
    pad_mode = cfg["data"].get("pad_mode", "post")
    trunc_mode = cfg["data"].get("trunc_mode", "post")

    X_list, ids = load_test_sequences(args.csv, id_col, feat_cols)

    ds = SeqTestDataset(X_list, pad_len, scaler, pad_mode, trunc_mode)
    pin_memory = torch.cuda.is_available()
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=pin_memory,
                    collate_fn=collate_pred)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RNNWithAttn(
        input_dim=len(feat_cols),
        num_classes=len(id2label),
        rnn_type=cfg["model"]["rnn_type"],
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        bidirectional=bool(cfg["model"]["bidirectional"]),
        input_dropout=float(cfg["model"]["input_dropout"]),
        attn_dropout=float(cfg["model"]["attn_dropout"]),
        fc_dropout=float(cfg["model"]["fc_dropout"]),
    ).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    all_probs, order = [], []
    with torch.inference_mode():
        for xb, L, batch_ids in dl:
            xb, L = xb.to(device), L.to(device)
            logits = model(xb, L)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            order.extend(batch_ids)

    P = np.concatenate(all_probs, axis=0)

    pred_idx = P.argmax(1)
    pred_labels = [id2label[int(i)] for i in pred_idx]

    out_df = pd.DataFrame({id_col: [ids[i] for i in order], label_col: pred_labels})
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  |  from run: {art_dir}")


if __name__ == "__main__":
    main()
