"""
Training / evaluation entry point for the MRAD reproduction.

Usage (synthetic demo, no external data needed):
    python train.py --anomaly trend --epochs 5
    python train.py --anomaly shapelet --window 128 --levels 4

Evaluation uses the point-wise best-F1 protocol described in the paper
(threshold chosen to maximize point-wise F1; NO point-adjustment by default).
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MRAD
from data import make_tods_like, WindowDataset, _normalize


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def best_point_f1(scores, labels, num_thresholds=200):
    """Search a threshold maximizing point-wise F1. Returns (f1, p, r, thr)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    qs = np.linspace(scores.min(), scores.max(), num_thresholds)
    best = (0.0, 0.0, 0.0, qs[0])
    P = labels.sum()
    for thr in qs:
        pred = scores > thr
        tp = np.logical_and(pred, labels == 1).sum()
        fp = np.logical_and(pred, labels == 0).sum()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (P + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best[0]:
            best = (f1, prec, rec, thr)
    return best


@torch.no_grad()
def score_series(model, series, window, device, batch_size=128):
    """Compute a per-timestamp anomaly score for a full (T, C) series by
    averaging window-level scores over all windows covering each timestamp."""
    model.eval()
    T = len(series)
    ds = WindowDataset(series, window, stride=window)  # non-overlap + tail window
    dl = DataLoader(ds, batch_size=batch_size)
    acc = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    starts = ds.starts
    si = 0
    for batch in dl:
        batch = batch.to(device)
        out = model(batch)
        s = model.anomaly_score(out).cpu().numpy()  # (B, window)
        for b in range(s.shape[0]):
            st = starts[si]
            acc[st:st + window] += s[b]
            cnt[st:st + window] += 1
            si += 1
    cnt[cnt == 0] = 1
    return acc / cnt


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(model, train_series, window, device, epochs=5, lr=1e-3,
          batch_size=64, lam=0.2, stride=None, log_every=1):
    stride = stride or window // 2
    ds = WindowDataset(train_series, window, stride=stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        tot, rec, const, nb = 0.0, 0.0, 0.0, 0
        for batch in dl:
            batch = batch.to(device)
            out = model(batch)
            loss, logs = model.total_loss(out, lam=lam)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); rec += logs["rec"]; const += logs["const"]; nb += 1
        if (ep + 1) % log_every == 0:
            print(f"  epoch {ep+1:3d} | loss {tot/nb:.4f} "
                  f"| rec {rec/nb:.4f} | const {const/nb:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anomaly", default="trend",
                    choices=["point_global", "point_contextual", "shapelet",
                             "seasonal", "trend"])
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--wavelet", default="db2", choices=["haar", "db2", "db4"])
    ap.add_argument("--n_sparse_levels", type=int, default=1)
    ap.add_argument("--sparse_radius", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | anomaly={args.anomaly} | window={args.window} "
          f"| levels={args.levels} | wavelet={args.wavelet}")

    train_raw, test_raw, label = make_tods_like(args.anomaly, seed=args.seed)
    train_n, test_n = _normalize(train_raw, test_raw)

    model = MRAD(
        in_dim=train_n.shape[1], d_model=args.d_model, levels=args.levels,
        wavelet=args.wavelet, n_heads=args.n_heads,
        n_sparse_levels=args.n_sparse_levels, sparse_radius=args.sparse_radius,
        alpha=args.alpha,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"#params = {n_params/1e3:.1f}K")

    train(model, train_n, args.window, device, epochs=args.epochs,
          lr=args.lr, lam=args.lam)

    scores = score_series(model, test_n, args.window, device)
    f1, p, r, thr = best_point_f1(scores, label)
    print(f"\n[RESULT] point-wise best-F1 = {f1:.4f} "
          f"(precision {p:.4f}, recall {r:.4f}, thr {thr:.4f})")


if __name__ == "__main__":
    main()
