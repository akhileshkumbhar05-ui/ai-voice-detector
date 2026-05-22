
# import os
# import argparse
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path
# from models.cnn_melspec import TinyMelCNN
# from utils.audio import load_audio, pad_or_trim, logmel, TARGET_SR
# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain

# AUG = Compose([
#     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
#     TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
#     PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
#     #Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.3),
#     Gain(min_gain_db=-3, max_gain_db=3, p=0.3),
# ])

# class FolderDataset(Dataset):
#     def __init__(self, root: str, split: str = "train", val_ratio: float = 0.15, seed: int = 42):
#         self.root = Path(root)
#         human = sorted((self.root / "human").glob("*.wav"))
#         ai = sorted((self.root / "ai").glob("*.wav"))
#         all_pairs = [(p, 0) for p in human] + [(p, 1) for p in ai]
#         random.Random(seed).shuffle(all_pairs)
#         n_val = int(len(all_pairs) * val_ratio)
#         if split == "train":
#             self.items = all_pairs[n_val:]
#         else:
#             self.items = all_pairs[:n_val]

#     def __len__(self): return len(self.items)

#     def __getitem__(self, idx):
#         path, label = self.items[idx]
#         y, sr = load_audio(str(path), TARGET_SR)
#         y = pad_or_trim(y, duration_s=3.0, sr=sr)
#         if self.__dict__.get("is_train", False):
#             from audiomentations import Compose
#             y = AUG(samples=y, sample_rate=sr)
#         mel = logmel(y, sr)
#         x = torch.from_numpy(mel).unsqueeze(0)
#         y_t = torch.tensor(label, dtype=torch.long)
#         return x, y_t

# def train(args):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     ds_tr = FolderDataset(args.data_dir, split="train"); ds_tr.is_train = True
#     ds_va = FolderDataset(args.data_dir, split="val"); ds_va.is_train = False
#     dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
#     dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)

#     model = TinyMelCNN().to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

#     best = 0.0
#     for epoch in range(args.epochs):
#         model.train()
#         tr_loss=0; tr_acc=0; n=0
#         for x,y in dl_tr:
#             x,y = x.to(device), y.to(device)
#             logits = model(x)
#             loss = torch.nn.functional.cross_entropy(logits, y)
#             opt.zero_grad(); loss.backward(); opt.step()
#             tr_loss += float(loss) * len(x)
#             tr_acc += float((logits.argmax(-1)==y).float().mean()) * len(x)
#             n += len(x)
#         tr_loss/=max(n,1); tr_acc/=max(n,1)

#         model.eval()
#         va_acc=0; m=0
#         with torch.no_grad():
#             for x,y in dl_va:
#                 x,y=x.to(device), y.to(device)
#                 logits = model(x)
#                 va_acc += float((logits.argmax(-1)==y).float().mean()) * len(x)
#                 m += len(x)
#         va_acc/=max(m,1)

#         print(f"epoch {epoch+1}/{args.epochs} | loss {tr_loss:.3f} | tr_acc {tr_acc:.3f} | va_acc {va_acc:.3f}")
#         if va_acc > best:
#             best = va_acc
#             os.makedirs(os.path.dirname(args.out), exist_ok=True)
#             torch.save(model.state_dict(), args.out)
#             print(f"saved best to {args.out} (acc={best:.3f})")

# if __name__ == '__main__':
#     p = argparse.ArgumentParser()
#     p.add_argument('--data_dir', type=str, required=True, help='Folder with subfolders human/ and ai/')
#     p.add_argument('--out', type=str, default='app/models/weights/cnn_melspec.pth')
#     p.add_argument('--epochs', type=int, default=5)
#     p.add_argument('--batch_size', type=int, default=32)
#     p.add_argument('--lr', type=float, default=1e-3)
#     args = p.parse_args()
#     train(args)


#------------------------------------------------------------

import os
import argparse
import random
from pathlib import Path
from contextlib import nullcontext
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

# ---------- Local imports ----------
try:
    from .models.cnn_melspec import TinyMelCNN
    from .utils.audio import load_audio, pad_or_trim, logmel, TARGET_SR
except ImportError:
    from app.models.cnn_melspec import TinyMelCNN
    from app.utils.audio import load_audio, pad_or_trim, logmel, TARGET_SR

# ---------- Augmentations (robust across versions) ----------
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, BandPassFilter
)

def make_gain(min_db, max_db, p):
    """Handle both min_gain_in_db/max_gain_in_db and min_gain_db/max_gain_db."""
    from audiomentations import Gain as _Gain
    try:
        return _Gain(min_gain_in_db=min_db, max_gain_in_db=max_db, p=p)
    except TypeError:
        return _Gain(min_gain_db=min_db, max_gain_db=max_db, p=p)

def make_clipping(p=0.3):
    """
    Build ClippingDistortion across versions.
    Newer:  min_percent/max_percent (0..20 typical)
    Older:  min_percentile_threshold/max_percentile_threshold in [0..100]
    Returns None if not available.
    """
    try:
        from audiomentations import ClippingDistortion as _Clip
    except Exception:
        return None

    # Try newer signature
    for kwargs in (
        dict(min_percent=0.0, max_percent=20.0, p=p),
        dict(min_percent=5.0,  max_percent=30.0, p=p),
    ):
        try:
            return _Clip(**kwargs)
        except Exception:
            pass

    # Try older signature
    for kwargs in (
        dict(min_percentile_threshold=95, max_percentile_threshold=100, p=p),
        dict(min_percentile_threshold=90, max_percentile_threshold=99,  p=p),
    ):
        try:
            return _Clip(**kwargs)
        except Exception:
            pass

    return None

def have_fast_mp3():
    return importlib.util.find_spec("fast_mp3_augment") is not None

def make_mp3_compression(min_bitrate=48, max_bitrate=96, p=0.6):
    """
    Only enable Mp3Compression when the fast backend is present.
    On Windows without the extra package this often breaks; we skip it.
    """
    if not have_fast_mp3():
        return None
    try:
        from audiomentations import Mp3Compression as _Mp3
        # Prefer the fast backend; if API lacks backend arg, constructor still works.
        try:
            return _Mp3(min_bitrate=min_bitrate, max_bitrate=max_bitrate, p=p, backend="fast_mp3_augment")
        except TypeError:
            return _Mp3(min_bitrate=min_bitrate, max_bitrate=max_bitrate, p=p)
    except Exception:
        return None

# ---------- Repro ----------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------- Dataset ----------
class FolderDataset(Dataset):
    """
    data_dir/
      human/*.wav
      ai/*.wav
    """
    def __init__(self, root: str, split: str = "train", val_ratio: float = 0.15,
                 seed: int = 42, clip_seconds: float = 3.0):
        self.root = Path(root)
        self.clip_seconds = float(clip_seconds)

        human = sorted((self.root / "human").glob("*.wav"))
        ai = sorted((self.root / "ai").glob("*.wav"))
        pairs = [(p, 0) for p in human] + [(p, 1) for p in ai]

        rng = random.Random(seed)
        rng.shuffle(pairs)

        n_val = int(len(pairs) * val_ratio)
        self.items = pairs[n_val:] if split == "train" else pairs[:n_val]
        self.is_train = split == "train"

        self._len_h = sum(1 for _, y in self.items if y == 0)
        self._len_a = sum(1 for _, y in self.items if y == 1)

        # Human: mild, natural perturbations
        self.aug_human = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.4),
            TimeStretch(min_rate=0.96, max_rate=1.04, p=0.3),
            PitchShift(min_semitones=-1, max_semitones=1, p=0.2),
            make_gain(-4, 4, p=0.3),
        ])

        # AI: replay-aware chain (speaker/room/mic simulation)
        ai_transforms = [
            BandPassFilter(min_center_freq=200.0, max_center_freq=3500.0, p=0.5),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.25),
            make_gain(-6, 6, p=0.3),
        ]
        clip = make_clipping(p=0.3)
        if clip is not None:
            ai_transforms.insert(1, clip)
        mp3 = make_mp3_compression()
        if mp3 is not None:
            ai_transforms.insert(0, mp3)

        self.aug_ai = Compose(ai_transforms)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        y, sr = load_audio(str(path), TARGET_SR)
        y = pad_or_trim(y, duration_s=self.clip_seconds, sr=sr)

        if self.is_train:
            if label == 1:
                y = self.aug_ai(samples=y, sample_rate=sr)
            else:
                y = self.aug_human(samples=y, sample_rate=sr)

        mel = logmel(y, sr)  # (n_mels, T)
        x = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)
        y_t = torch.tensor(label, dtype=torch.long)
        return x, y_t

# ---------- Dataloaders ----------
def make_dataloaders(args):
    ds_tr = FolderDataset(args.data_dir, split="train", val_ratio=args.val_ratio,
                          seed=args.seed, clip_seconds=args.clip_seconds)
    ds_va = FolderDataset(args.data_dir, split="val", val_ratio=args.val_ratio,
                          seed=args.seed, clip_seconds=args.clip_seconds)

    # Windows is happier with workers=0; keep configurable
    workers = args.workers if args.workers >= 0 else (0 if os.name == "nt" else max(1, (os.cpu_count() or 4)//2))
    pin = (not args.cpu) and torch.cuda.is_available()

    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True,
        num_workers=workers, pin_memory=pin,
        persistent_workers=(workers > 0), drop_last=True,
    )
    dl_va = DataLoader(
        ds_va, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=workers, pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    return ds_tr, ds_va, dl_tr, dl_va

def class_weights_from_dataset(ds: FolderDataset, eps: float = 1e-6):
    n_h, n_a = max(ds._len_h, eps), max(ds._len_a, eps)
    w_h = (n_h + n_a) / (2 * n_h)
    w_a = (n_h + n_a) / (2 * n_a)
    return torch.tensor([w_h, w_a], dtype=torch.float32)

# ---------- Training / Eval ----------
def train_one_epoch(model, dl, device, opt, scaler, autocast_ctx, loss_fn, grad_accum=1):
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    opt.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(dl):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(x)
            loss = loss_fn(logits, y)

        loss = loss / grad_accum
        if getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if getattr(scaler, "is_enabled", lambda: False)():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        total_loss += float(loss) * x.size(0) * grad_accum
        correct += int((logits.argmax(1) == y).sum().item())
        seen += x.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)

@torch.no_grad()
def evaluate(model, dl, device, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss) * x.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        seen += x.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)

def main(args):
    set_seed(args.seed)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    cudnn.benchmark = True

    ds_tr, ds_va, dl_tr, dl_va = make_dataloaders(args)
    print(f"Train items: {len(ds_tr)} (human={ds_tr._len_h}, ai={ds_tr._len_a})")
    print(f"Val   items: {len(ds_va)}")

    model = TinyMelCNN().to(device)

    weights = class_weights_from_dataset(ds_tr).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # AMP (use new torch.amp if available, else fallback)
    try:
        from torch.amp import GradScaler, autocast as amp_autocast
        scaler = GradScaler("cuda", enabled=(device == "cuda" and args.amp))
        autocast_ctx = amp_autocast("cuda") if (device == "cuda" and args.amp) else nullcontext()
    except Exception:
        from torch.cuda.amp import GradScaler, autocast as amp_autocast  # deprecated but works
        scaler = GradScaler(enabled=(device == "cuda" and args.amp))
        autocast_ctx = amp_autocast() if (device == "cuda" and args.amp) else nullcontext()

    best_va = -1.0
    patience_counter = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, dl_tr, device, opt, scaler, autocast_ctx, loss_fn,
            grad_accum=args.grad_accum
        )
        va_loss, va_acc = evaluate(model, dl_va, device, loss_fn)

        print(f"epoch {epoch+1:02d}/{args.epochs} | train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f}")

        # Save "last" every epoch
        torch.save(model.state_dict(), args.out.replace(".pth", ".last.pth"))

        if va_acc > best_va + 1e-4:
            best_va = va_acc
            torch.save(model.state_dict(), args.out)
            patience_counter = 0
            print(f"✅ Saved best to {args.out} (val_acc={best_va:.3f})")
        else:
            patience_counter += 1
            if args.early_stop > 0 and patience_counter >= args.early_stop:
                print(f"⏹️ Early stopping at epoch {epoch+1} (best val_acc={best_va:.3f})")
                break

    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train AI Voice Detector (replay-aware, version-robust, no fast_mp3 required)")
    p.add_argument("--data_dir", type=str, required=True, help="Folder with subfolders human/ and ai/")
    p.add_argument("--out", type=str, default="app/models/weights/cnn_melspec.pth")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--clip_seconds", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=-1)   # try --workers 0 on Windows if you see issues
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--early_stop", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)