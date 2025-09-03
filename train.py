import os, json, numpy as np, pandas as pd
from typing import List, Tuple, Dict, Any, Literal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hydra
import joblib

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PadMode = Literal["post", "pre", "center"]

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_dataframe(csv_path: str, id_col: str, label_col: str):
    df = pd.read_csv(csv_path)

    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(f'Missing required columns: {id_col}/{label_col}')

    feat_cols = [
        c for c in df.columns
        if c not in (id_col, label_col) and pd.api.types.is_numeric_dtype(df[c])
    ]

    df = df[[id_col, label_col] + feat_cols].copy()

    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    return df, feat_cols

def build_sequences(df: pd.DataFrame, id_col: str, 
                    label_col: str, feat_cols: List[str]) -> Tuple[List[np.ndarray], List, List]:
    
    X, y, ids = [], [], []

    for sid, g in df.groupby(id_col, sort=False):
        arr = g[feat_cols].to_numpy(dtype=np.float32)
        lab = g[label_col].iloc[-1]

        X.append(arr) 
        y.append(lab)
        ids.append(sid)

    return X, y, ids

def encode_labels(y_raw: List[str]):
    arr = np.asarray(y_raw)
    codes, uniques = pd.factorize(arr)

    id2label = {int(i): str(v) for i, v in enumerate(uniques.tolist())}
    label2id = {v: i for i, v in id2label.items()}

    return codes.astype(int).tolist(), id2label, label2id

def pad_to_len(X: np.ndarray, pad_len: int, pad_mode: PadMode = "post", 
               trunc_mode: PadMode = "post", value: float = 0.0):

    X = np.asarray(X)
    T, C = X.shape
    if T == pad_len:
        return np.ascontiguousarray(X.astype(np.float32))
    
    if T > pad_len:
        if trunc_mode == 'post':
            out = X[:pad_len]
        elif trunc_mode == 'pre':
            out = X[-pad_len:]
        elif trunc_mode == 'center':
            start = (T - pad_len) // 2
            out = X[start: start + pad_len]
        else:
            raise ValueError('Trunc mode is bad')

        return np.ascontiguousarray(out.astype(np.float32))
    
    pad_total = pad_len - T
    if pad_mode == 'post':
        pad_before, pad_after = 0, pad_total
    elif pad_mode == 'pre':
        pad_before, pad_after = pad_total, 0
    elif pad_mode == 'center':
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
    else:
        raise ValueError('Pad mode is bad')

    pad_width = ((pad_before, pad_after), (0, 0))
    out = np.pad(X, pad_width=pad_width, mode='constant', constant_values=value)
    return np.ascontiguousarray(out.astype(np.float32))

class SeqDataset(Dataset):
    def __init__(self, X: List[np.ndarray], y: List[int],
                pad_len: int, scaler: StandardScaler, 
                pad_mode: PadMode ='post', trunc_mode: PadMode ='post'):

        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.pad_len = int(pad_len)
        self.scaler = scaler
        self.pad_mode = pad_mode
        self.trunc_mode = trunc_mode

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        x = self.scaler.transform(self.X[idx])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        L = x.shape[0]
        x = pad_to_len(x, self.pad_len, self.pad_mode, self.trunc_mode)

        return torch.from_numpy(x), torch.tensor(int(self.y[idx])), int(min(L, self.pad_len)), idx

def collate_fn(batch):
    xs, ys, lens, ids = zip(*batch)
    xs = torch.stack(xs, 0)                 
    ys = torch.stack(ys, 0).long()
    lens = torch.tensor(lens).long()
    return xs, ys, lens, ids

class AttnPool(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        
        self.proj = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, mask=None):
        A = torch.tanh(self.proj(H))
        e = self.v(A).squeeze(-1)

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
        rnn = nn.LSTM if rnn_type.lower()=="lstm" else nn.GRU
        self.rnn = rnn(input_dim, hidden_size, num_layers,
                       batch_first=True, bidirectional=bidirectional,
                       dropout=0.0 if num_layers==1 else 0.1)
        
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
        mask = idx < lengths.unsqueeze(1)
        ctx, attn = self.attn(H, mask=mask)

        return self.head(ctx), attn


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval()
    tot, ys, probs = 0.0, [], []

    for xb, yb, L, _ in loader:
        xb, yb, L = xb.to(device), yb.to(device), L.to(device)
        logits, _ = model(xb, L)
        loss = criterion(logits, yb)
        tot += float(loss.item()) * xb.size(0)
        probs.append(F.softmax(logits, -1).cpu().numpy())
        ys.append(yb.cpu().numpy())

    y = np.concatenate(ys); p = np.concatenate(probs)
    from sklearn.metrics import f1_score

    return tot/len(loader.dataset), float(f1_score(y, p.argmax(1), average="macro")), p

def train_one_epoch(model, loader, optimizer, device, criterion, scaler=None, grad_clip=1.0, scheduler=None):
    model.train()
    tot, ys, probs = 0.0, [], []
    use_amp = False # (scaler is not None) and (torch.cuda.is_available())

    for xb, yb, L, _ in loader:
        xb, yb, L = xb.to(device), yb.to(device), L.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(xb, L)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(xb, L)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        tot += float(loss.item()) * xb.size(0)
        probs.append(F.softmax(logits.detach().cpu(), -1).numpy())
        ys.append(yb.cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    from sklearn.metrics import f1_score

    y = np.concatenate(ys)
    p = np.concatenate(probs)

    return tot/len(loader.dataset), float(f1_score(y, p.argmax(1), average="macro"))

def save_checkpoint(out_dir: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler, meta: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "meta": meta}, os.path.join(out_dir, "checkpoint.pth"))
    
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.get("seed", 42)))

    df, feat_cols = load_dataframe(cfg.data.train_csv, cfg.data.id_col, cfg.data.label_col)
    X_list, y_raw, ids = build_sequences(df, cfg.data.id_col, cfg.data.label_col, feat_cols)
    y_int, id2label, label2id = encode_labels(y_raw)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_list, y_int, test_size=float(cfg.data.val_size),
        shuffle=bool(cfg.data.shuffle), stratify=y_int, random_state=int(cfg.seed))

    pad_len = max(16, int(np.percentile([len(s) for s in X_tr], int(cfg.data.pad_percentile))))
    scaler = StandardScaler().fit(np.vstack(X_tr))

    pin_memory = True if torch.cuda.is_available() else False

    tr_ds = SeqDataset(X_tr, y_tr, pad_len, scaler, cfg.data.get("pad_mode","post"), cfg.data.get("trunc_mode","post"))
    va_ds = SeqDataset(X_va, y_va, pad_len, scaler, cfg.data.get("pad_mode","post"), cfg.data.get("trunc_mode","post"))
    tr_ld = DataLoader(tr_ds, batch_size=int(cfg.trainer.batch_size), shuffle=True,
                       num_workers=int(cfg.trainer.num_workers), pin_memory=pin_memory, collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=int(cfg.trainer.batch_size), shuffle=False,
                       num_workers=int(cfg.trainer.num_workers), pin_memory=pin_memory, collate_fn=collate_fn)

    device = "cuda" if (torch.cuda.is_available() and bool(cfg.trainer.get("use_cuda", True))) else "cpu"

    model = RNNWithAttn(
        input_dim=len(feat_cols), num_classes=len(id2label),
        rnn_type=str(cfg.model.rnn_type), hidden_size=int(cfg.model.hidden_size),
        num_layers=int(cfg.model.num_layers), bidirectional=bool(cfg.model.bidirectional),
        input_dropout=float(cfg.model.input_dropout), attn_dropout=float(cfg.model.attn_dropout),
        fc_dropout=float(cfg.model.fc_dropout)
    ).to(device)

    w = None
    if cfg.trainer.get("class_weight"): w = torch.tensor(list(cfg.trainer.class_weight), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.trainer.lr), weight_decay=float(cfg.trainer.weight_decay))
    scaler_amp = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_f1, best_state, patience = -1.0, None, int(cfg.trainer.early_stop_patience)
    out_dir = HydraConfig.get().runtime.output_dir
    for epoch in range(1, int(cfg.trainer.epochs)+1):
        scaler_to_pass = scaler_amp if device == "cuda" else None
        tr_loss, tr_f1 = train_one_epoch(model, tr_ld, optimizer, device, criterion, scaler=scaler_to_pass, grad_clip=float(cfg.trainer.grad_clip))
        va_loss, va_f1, va_probs = evaluate(model, va_ld, device, criterion)

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_f1:.4f} | val {va_loss:.4f}/{va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1, best_state, patience = va_f1, {k: v.detach().cpu() for k,v in model.state_dict().items()}, int(cfg.trainer.early_stop_patience)
            meta = {"epoch": epoch, "val_f1": float(best_f1), "pad_len": pad_len, "feat_cols": feat_cols}
            save_checkpoint(out_dir, model, optimizer, scaler_amp, meta)
            joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
            with open(os.path.join(out_dir, "id2label.json"), "w") as f: json.dump(id2label, f, indent=2)
            with open(os.path.join(out_dir, "config_composed.yaml"), "w") as f: f.write(OmegaConf.to_yaml(cfg))
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping."); break

    torch.save(best_state if best_state is not None else model.state_dict(),
               os.path.join(out_dir, "model_state_dict.pt"))
    print(f"Done. Best val F1: {best_f1:.4f}. Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
