"""
MRAD: Multi-Resolution Anomaly Detection
A faithful, from-scratch reproduction of the method described in
"Beyond Uniform Time-Frequency Projections: A Novel Approach for Time Series
Anomaly Detection" (MRAD).

This file implements the three core components from the paper:
  1. Approximate Orthogonal Decomposition  (learnable DWT, Eq. 15-17)
  2. Adjacent Resolution Fusion            (CRA / SCRA, Eq. 21-25)
  3. Reconstruction-based Anomaly Scoring  (Expand + score, Eq. 26-29)

Design choices where the paper is under-specified are marked with [CHOICE].
They are the natural reading of the equations; change them to match your own
implementation if needed.

Notation map (paper -> code)
  X (T x d)        -> input x, shape (B, T, in_dim)
  h^{(j)}, g^{(j)} -> per-level learnable low/high pass filters
  a_j, d_j         -> approx_list[j-1], detail_list[j-1]
  hat_a_j          -> refined approximation after CRA fusion
  alpha            -> decay factor in the multi-level loss / score
  lambda           -> weight of the orthogonality regularizer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Classical orthonormal scaling filters used as initialization (||h||_2 = 1).
# g is *derived* from h at runtime via the quadrature-mirror relationship,
# so we only store the low-pass taps.
# ----------------------------------------------------------------------------
WAVELET_FILTERS = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db2": [
        0.48296291314469025,
        0.836516303737469,
        0.22414386804185735,
        -0.12940952255092145,
    ],
    "db4": [
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ],
}


def _safe_pad(x, pl, pr):
    """Reflect-pad along the last dim, falling back to circular padding when the
    requested pad is too large for reflect (happens only at very deep levels)."""
    pad = max(pl, pr)
    if pad == 0:
        return x
    if pad < x.shape[-1]:
        return F.pad(x, (pl, pr), mode="reflect")
    return F.pad(x, (pl, pr), mode="circular")


class FeedForward(nn.Module):
    """Position-wise FFN used inside the decomposition and the fusion blocks."""

    def __init__(self, d_model, hidden=None, dropout=0.0):
        super().__init__()
        hidden = hidden or 2 * d_model  # [CHOICE] paper does not fix the width
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 1. Approximate Orthogonal Decomposition  (learnable DWT)
# ============================================================================
class LearnableDWT(nn.Module):
    """Learnable multi-level wavelet decomposition.

    Each level j:
        a_in = FeedForward(a_{j-1})
        a_j  = (a_in * h^{(j)}) downsample_2          (Eq. 15)
        d_j  = (a_in * g^{(j)}) downsample_2
        g^{(j)}[n] = (-1)^n h^{(j)}[L-1-n]            (Eq. 17, strict cross-orth.)
        ||h^{(j)}||_2 = 1                             (Eq. 18, energy norm)

    Filters are initialized from a classical orthonormal basis and shared across
    feature channels (depthwise conv), matching "apply the same wavelet to each
    channel" while keeping the transform a genuine MRA.
    """

    def __init__(self, d_model, levels, wavelet="db2", ffn_dropout=0.0):
        super().__init__()
        h0 = torch.tensor(WAVELET_FILTERS[wavelet], dtype=torch.float32)
        self.L = h0.numel()
        self.levels = levels
        self.d_model = d_model
        # per-level learnable low-pass filters h^{(j)}
        self.h = nn.ParameterList(
            [nn.Parameter(h0.clone()) for _ in range(levels)]
        )
        self.ffn = nn.ModuleList(
            [FeedForward(d_model, dropout=ffn_dropout) for _ in range(levels)]
        )

    def filters(self, j):
        """Return (h, g) for level j (0-indexed). h is energy-normalized; g is the
        quadrature mirror of h. Normalizing in forward == projection h<-h/||h||."""
        h = self.h[j]
        h = h / (h.norm() + 1e-12)
        L = self.L
        idx = torch.arange(L, device=h.device)
        sign = torch.where(idx % 2 == 0, 1.0, -1.0)
        g = sign * h.flip(0)  # g[n] = (-1)^n h[L-1-n]
        return h, g

    def _conv_down(self, x, filt):
        """x: (B, C, T) -> (B, C, T//2). Depthwise conv + stride-2 downsample."""
        B, C, T = x.shape
        L = filt.numel()
        pad = L - 2  # gives output length exactly T//2 for even T
        if pad > 0:
            pl, pr = pad // 2, pad - pad // 2
            x = _safe_pad(x, pl, pr)
        kernel = filt.view(1, 1, L).expand(C, 1, L)
        return F.conv1d(x, kernel, stride=2, groups=C)

    def forward(self, z):
        """z: (B, T, C). Returns approx_list=[a_1..a_K], detail_list=[d_1..d_K]."""
        a = z
        approx_list, detail_list = [], []
        for j in range(self.levels):
            a_in = self.ffn[j](a).transpose(1, 2)  # (B, C, T_j-1)
            h, g = self.filters(j)
            a_next = self._conv_down(a_in, h).transpose(1, 2)  # (B, T_j, C)
            d_next = self._conv_down(a_in, g).transpose(1, 2)
            approx_list.append(a_next)
            detail_list.append(d_next)
            a = a_next
        return approx_list, detail_list


# ============================================================================
# 2. Adjacent Resolution Fusion (Cross-Resolution Attention)
# ============================================================================
class LearnableUpsample(nn.Module):
    """'Inverse convolutional upsampling' (Eq. 21): T -> 2T via depthwise
    transposed convolution, used to align hat_a_{j+1} to the resolution of d_j."""

    def __init__(self, d_model, kernel=4):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=kernel, stride=2,
            padding=(kernel - 2) // 2, groups=d_model,
        )

    def forward(self, x):  # (B, T, C) -> (B, 2T, C)
        return self.up(x.transpose(1, 2)).transpose(1, 2)


def local_mask(Tq, Tk, radius, device):
    """SCRA mask (Eq. 25): M[i,j] = 1 iff |floor(i/2) - j| <= radius.
    i indexes the (longer) query at level j, j indexes the key at level j+1."""
    qi = torch.arange(Tq, device=device).view(Tq, 1)
    ki = torch.arange(Tk, device=device).view(1, Tk)
    return (torch.div(qi, 2, rounding_mode="floor") - ki).abs() <= radius


class CrossResolutionAttention(nn.Module):
    """CRA / SCRA (Eq. 22, 24).

    Query comes from level j (length T/2^j); Key/Value from level j+1
    (length T/2^{j+1}). The attention map is therefore *rectangular* and the
    output has the same shape as the Query. A local mask turns CRA into SCRA.

    [CHOICE] Multi-head is standard practice; set n_heads=1 to match the single
    projection in the paper's equation exactly. The mask is applied as an
    additive -inf before softmax (the paper writes a Hadamard product, but a 0/1
    multiply before softmax does not actually exclude positions; masking is the
    clear intent of "force attention onto local ranges").
    """

    def __init__(self, d_model, n_heads=4, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_seq, kv_seq, mask=None):
        B, Tq, C = q_seq.shape
        Tk = kv_seq.shape[1]
        q = self.Wq(q_seq).view(B, Tq, self.h, self.dh).transpose(1, 2)
        k = self.Wk(kv_seq).view(B, Tk, self.h, self.dh).transpose(1, 2)
        v = self.Wv(kv_seq).view(B, Tk, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)  # (B,h,Tq,Tk)
        if mask is not None:
            scores = scores.masked_fill(~mask.view(1, 1, Tq, Tk), float("-inf"))
        attn = self.drop(scores.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, Tq, C)
        return self.Wo(out)


# ============================================================================
# Full MRAD model
# ============================================================================
class MRAD(nn.Module):
    def __init__(
        self,
        in_dim,
        d_model=64,
        levels=4,
        wavelet="db2",
        n_heads=4,
        n_sparse_levels=1,   # the N lowest (highest-frequency) levels use SCRA
        sparse_radius=3,     # window radius r of SCRA
        alpha=0.5,           # decay factor for multi-level loss / score
        dropout=0.0,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.levels = levels
        self.alpha = alpha
        self.n_sparse_levels = n_sparse_levels
        self.sparse_radius = sparse_radius
        self.norm_eps = norm_eps

        self.embed = nn.Linear(in_dim, d_model)
        self.dwt = LearnableDWT(d_model, levels, wavelet, ffn_dropout=dropout)

        # fusion blocks, indexed by (level-1) for j = 1 .. K-1
        self.up = nn.ModuleList([LearnableUpsample(d_model) for _ in range(levels)])
        self.cra = nn.ModuleList(
            [CrossResolutionAttention(d_model, n_heads, dropout) for _ in range(levels)]
        )
        self.fuse_ffn = nn.ModuleList([FeedForward(d_model, dropout=dropout) for _ in range(levels)])
        self.norm_z = nn.ModuleList([nn.LayerNorm(d_model, eps=norm_eps) for _ in range(levels)])
        self.norm_a = nn.ModuleList([nn.LayerNorm(d_model, eps=norm_eps) for _ in range(levels)])

        # final synthesis (Eq. 26): IDWT with time-reversed level-1 filters
        self.final_ffn = FeedForward(d_model, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.proj_out = nn.Linear(d_model, in_dim)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _zero_upsample(x):  # (B, C, T) -> (B, C, 2T) by inserting zeros
        B, C, T = x.shape
        out = x.new_zeros(B, C, 2 * T)
        out[:, :, ::2] = x
        return out

    def _conv_same(self, x, filt):  # depthwise conv, length-preserving
        B, C, T = x.shape
        L = filt.numel()
        pad = L - 1
        pl, pr = pad // 2, pad - pad // 2
        x = _safe_pad(x, pl, pr)
        kernel = filt.view(1, 1, L).expand(C, 1, L)
        return F.conv1d(x, kernel, groups=C)

    # -- forward -------------------------------------------------------------
    def forward(self, x):
        """x: (B, T, in_dim). Returns a dict with everything needed for loss/score."""
        B, T, _ = x.shape
        assert T % (2 ** self.levels) == 0, (
            f"window length {T} must be divisible by 2^levels = {2**self.levels}"
        )

        # --- Instance normalization (Eq. 14) ---
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        xn = (x - mu) / torch.sqrt(var + self.norm_eps)

        z = self.embed(xn)                              # (B, T, d_model)
        approx, detail = self.dwt(z)                    # a_1..a_K, d_1..d_K
        K = self.levels

        # --- Adjacent resolution fusion (top-down) ---
        hat_a = [None] * (K + 1)                        # hat_a[j], j=1..K
        hat_a[K] = approx[K - 1]                         # hat_a_K = a_K
        for j in range(K - 1, 0, -1):
            dj = detail[j - 1]                          # query, len T/2^j
            djp1 = detail[j]                            # key/value, len T/2^{j+1}
            up_a = self.up[j - 1](hat_a[j + 1])         # T/2^{j+1} -> T/2^j
            mask = None
            if j <= self.n_sparse_levels:               # high-frequency -> SCRA
                mask = local_mask(dj.shape[1], djp1.shape[1], self.sparse_radius, dj.device)
            cra_out = self.cra[j - 1](dj, djp1, mask)
            zj = self.norm_z[j - 1](cra_out + up_a)
            hat_a[j] = self.norm_a[j - 1](self.fuse_ffn[j - 1](zj) + zj)

        # --- Final reconstruction (Eq. 26): IDWT from hat_a_1 and d_1 ---
        h1, g1 = self.dwt.filters(0)
        a1 = hat_a[1].transpose(1, 2)                   # (B, C, T/2)
        d1 = detail[0].transpose(1, 2)
        rec = (
            self._conv_same(self._zero_upsample(a1), h1.flip(0))
            + self._conv_same(self._zero_upsample(d1), g1.flip(0))
        ).transpose(1, 2)                               # (B, T, d_model)
        rec = self.final_norm(self.final_ffn(rec))      # paper: Norm(FFN(.))
        x_hat = self.proj_out(rec)                      # (B, T, in_dim), in xn space

        return {
            "xn": xn, "x_hat": x_hat, "mu": mu, "var": var,
            "approx": approx, "detail": detail, "hat_a": hat_a,
        }

    # -- losses & scoring ----------------------------------------------------
    def orthogonality_loss(self):
        """L_const (Eq. 31): sum_k ( sum_n h[n]h[n-2k] - delta[k] )^2 over levels."""
        loss = x = None
        total = 0.0
        for j in range(self.levels):
            h, _ = self.dwt.filters(j)
            L = h.numel()
            lvl = 0.0
            for k in range(0, L // 2 + 1):
                shift = 2 * k
                if shift >= L:
                    break
                r = (h[: L - shift] * h[shift:]).sum()
                target = 1.0 if k == 0 else 0.0
                lvl = lvl + (r - target) ** 2
            total = total + lvl
        return total

    def reconstruction_loss(self, out):
        """L_rec (Eq. 30): ||X - X_hat||_2 + sum_j alpha^j ||a_j - hat_a_j||_1."""
        diff = out["xn"] - out["x_hat"]
        l2 = diff.norm(dim=-1).mean()                   # L2 per point, averaged
        rec = l2
        for j in range(1, self.levels + 1):
            aj = out["approx"][j - 1]
            haj = out["hat_a"][j]
            l1 = (aj - haj).abs().sum(dim=-1).mean()
            rec = rec + (self.alpha ** j) * l1
        return rec

    def total_loss(self, out, lam=0.2):
        rec = self.reconstruction_loss(out)
        const = self.orthogonality_loss()
        return rec + lam * const, {"rec": rec.item(), "const": float(const.detach())}

    @torch.no_grad()
    def anomaly_score(self, out):
        """Per-point score S(t) (Eq. 29):
        ||X(t)-X_hat(t)||_2 + sum_j alpha^j ||Expand(a_j)(t) - Expand(hat_a_j)(t)||_1
        Returns (B, T)."""
        T = out["xn"].shape[1]
        s = (out["xn"] - out["x_hat"]).norm(dim=-1)     # (B, T)
        for j in range(1, self.levels + 1):
            factor = 2 ** j
            Aj = out["approx"][j - 1].repeat_interleave(factor, dim=1)[:, :T]
            hAj = out["hat_a"][j].repeat_interleave(factor, dim=1)[:, :T]
            sj = (Aj - hAj).abs().sum(dim=-1)           # (B, T)
            s = s + (self.alpha ** j) * sj
        return s
