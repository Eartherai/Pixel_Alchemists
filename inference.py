#!/usr/bin/env python3

"""
kla_core.py - shared core for the KLA PS-1 restoration pipeline.

SINGLE SOURCE OF TRUTH. The training notebook, train.py and inference.py all
execute this exact text, so the network that is trained and the network that
KLA benchmarks are identical by construction.

Why this file exists: in the previous submission the training notebook used
res_scale=0.1 while the exported inference.py hard-coded 0.2. res_scale is not
a learned weight, so the exported script rebuilt a DIFFERENT function from the
same weights and scored 17.07 dB where the notebook measured 28.21 dB. Every
architecture constant now lives here and is additionally persisted inside the
checkpoint, so the two can never drift again.
"""
import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================== VST =========


class VST:
    """Variance-stabilising transform.

    The measured noise law is Var(x | mu) = sg^2 + ss^2 * mu^2 (additive
    Gaussian plus multiplicative speckle). Solving f'(mu) = 1/sqrt(Var(mu))
    gives f(x) = asinh(ss*x/sg)/ss, which makes the residual variance flat in
    brightness, so the network solves ONE denoising problem instead of a
    different one at every intensity.

    asinh is odd and finite on negatives, so the out-of-[0,1] NoisyLR values
    that the problem statement calls intentional pass through losslessly.
    log1p(relu(x)) would collapse every negative onto the same value.
    """

    def __init__(self, sg, ss, enabled=True):
        self.sg = max(float(sg), 1e-4)
        self.ss = max(float(ss), 1e-4)
        self.enabled = bool(enabled)
        self.norm = float(np.arcsinh(self.ss / self.sg) / self.ss)

    def __call__(self, x):
        if not self.enabled:
            return x
        return torch.asinh(x * (self.ss / self.sg)) / (self.ss * self.norm)

    def inverse(self, y):
        if not self.enabled:
            return y
        return torch.sinh(y * self.ss * self.norm) * (self.sg / self.ss)


# ============================================================ blocks ========


class ECA(nn.Module):
    """Efficient channel attention (Wang et al. 2020). ~k params, no FC layer."""

    def __init__(self, c, k=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)
        y = self.conv(y.squeeze(-1).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        return x * torch.sigmoid(y)


class ResBlock(nn.Module):
    """EDSR-style residual block + ECA. Plain 3x3 convs keep the tensor cores
    busy, which matters because throughput is a scored axis."""

    def __init__(self, c, res_scale=0.2):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.eca = ECA(c)
        self.res_scale = float(res_scale)

    def forward(self, x):
        return x + self.res_scale * self.eca(self.c2(F.silu(self.c1(x), inplace=True)))


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """NAFNet block (Chen et al., ECCV 2022) - activation-free, gated, with
    simplified channel attention. Higher quality per FLOP than a plain residual
    block; kept as a selectable alternative for the capacity study."""

    def __init__(self, c, dw_expand=2, ffn_expand=2, res_scale=1.0):
        super().__init__()
        d = c * dw_expand
        f = c * ffn_expand
        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, d, 1)
        self.conv2 = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.conv3 = nn.Conv2d(d // 2, c, 1)
        self.conv4 = nn.Conv2d(c, f, 1)
        self.conv5 = nn.Conv2d(f // 2, c, 1)
        # zero-init gates -> the block starts as the identity, so adding depth
        # can never make step 0 worse.
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.conv1(self.norm1(x))
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        y = self.conv4(self.norm2(x))
        y = self.sg(y)
        y = self.conv5(y)
        return x + y * self.gamma


# =========================================================== network ========


class RestoreNet(nn.Module):
    """VST -> stem -> N blocks -> PixelShuffle x2 -> + bilinear skip
       -> optional measurement-consistency (unrolling) correction.

    Two structural guarantees:
      * the tail is zero-initialised, so the step-0 output is EXACTLY the
        bilinear baseline and training can only improve on it;
      * the data-consistency head is zero-initialised too, so it can only help.

    Data consistency: we have MEASURED the forward operator D (the 6x6
    resampling kernel recovered by least squares). Given a first estimate
    x_hat, the measurement residual r = y - D(x_hat) says where the estimate
    disagrees with the observation. Feeding a learned map of r back into the
    output is one step of algorithm unrolling (Monga et al., IEEE SPM 2021,
    a reference KLA themselves circulated). It costs one 3x3 conv at LR
    resolution.
    """

    def __init__(self, width=96, blocks=20, scale=2, vst=None, res_scale=0.2,
                 block="res", dc=True, K=None, aux_head=False):
        super().__init__()
        self.scale = int(scale)
        self.vst = vst
        self.block_type = block
        # Auxiliary clean-LR head. Because D is MEASURED, the intermediate target
        # that normally does not exist -- a clean low-res image D(GT) -- can be
        # constructed. Supervising it gives the denoise-then-super-resolve
        # inductive bias inside a single network, and it is TRAINING ONLY: the
        # head is never evaluated at inference, so it costs zero scored latency.
        self.use_aux = bool(aux_head)
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        if block == "naf":
            body = [NAFBlock(width) for _ in range(blocks)]
        else:
            body = [ResBlock(width, res_scale) for _ in range(blocks)]
        self.body = nn.Sequential(*body)
        self.fuse = nn.Conv2d(width, width, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(width, width * self.scale * self.scale, 3, padding=1),
            nn.PixelShuffle(self.scale),
        )
        self.tail = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        if self.use_aux:
            self.aux = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(self.aux.weight)
            nn.init.zeros_(self.aux.bias)

        self.use_dc = bool(dc) and K is not None
        if self.use_dc:
            Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32)[None, None]
            self.register_buffer("Kb", Kt)
            self.support = int(Kt.shape[-1])
            self.off = self.support // 2 - 1
            self.dc = nn.Conv2d(1, self.scale * self.scale, 3, padding=1)
            nn.init.zeros_(self.dc.weight)
            nn.init.zeros_(self.dc.bias)
            self.dc_ps = nn.PixelShuffle(self.scale)

    def _down(self, x):
        """Apply the measured forward operator D: HR -> LR."""
        n0, n1 = x.shape[-2] // 2, x.shape[-1] // 2
        p = self.support
        K = self.Kb.to(x.dtype)
        y = F.conv2d(F.pad(x, (p, p, p, p), mode="reflect"), K)
        s = p - self.off
        return y[:, :, s::2, s::2][:, :, :n0, :n1]

    def forward(self, x, clamp=False, return_aux=False):
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear",
                             align_corners=False)
        h = self.vst(x) if self.vst is not None else x
        f = self.intro(h)
        f = self.fuse(self.body(f)) + f
        out = base + self.tail(self.up(f))
        if self.use_dc:
            r = x - self._down(out)
            out = out + self.dc_ps(self.dc(r))
        if clamp:
            out = out.clamp(0, 1)
        if return_aux and self.use_aux:
            # predicted clean LR = raw LR + zero-initialised correction
            return out, x + self.aux(f)
        return out


class BilinearNet(nn.Module):
    """Required baseline."""

    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, clamp=False):
        o = F.interpolate(x, scale_factor=self.scale, mode="bilinear",
                          align_corners=False)
        return o.clamp(0, 1) if clamp else o


class BicubicNet(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, clamp=False):
        o = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                          align_corners=False)
        return o.clamp(0, 1) if clamp else o


def build_net(cfg, calib, device="cpu"):
    """Construct the network from a config dict + calibration dict.

    Every architecture constant is read from cfg, and cfg is saved inside the
    checkpoint, so inference rebuilds precisely the trained function.
    """
    arch = cfg.get("arch", "main")
    if arch == "bilinear":
        return BilinearNet().to(device)
    if arch == "bicubic":
        return BicubicNet().to(device)
    vst = VST(calib["sigma_g"], calib["sigma_s"], enabled=cfg.get("use_vst", True))
    if arch == "compact":
        w, b = cfg.get("compact_width", 48), cfg.get("compact_blocks", 8)
    else:
        w, b = cfg.get("width", 96), cfg.get("blocks", 20)
    return RestoreNet(
        width=w, blocks=b, scale=2, vst=vst,
        res_scale=cfg.get("res_scale", 0.2),
        block=cfg.get("block", "res"),
        dc=cfg.get("dc_refine", True),
        K=calib.get("K"),
        aux_head=cfg.get("aux_head", False),
    ).to(device)


def downsample_with(x, K, support=None):
    """Apply a measured (or synthetic) resampling operator D: HR -> LR.

    Used to build the clean-LR target D(GT) for the auxiliary head, and to
    generate operator-shifted degradations for the robustness table.
    """
    Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32)
    if Kt.ndim == 2:
        Kt = Kt[None, None]
    s = support or Kt.shape[-1]
    off = s // 2 - 1
    n0, n1 = x.shape[-2] // 2, x.shape[-1] // 2
    y = F.conv2d(F.pad(x, (s, s, s, s), mode="reflect"), Kt.to(x.device, x.dtype))
    t = s - off
    return y[:, :, t::2, t::2][:, :, :n0, :n1]


def kernel_box(support=6):
    """Textbook 2x2 area-average kernel (operator-shift probe)."""
    o = support // 2 - 1
    k = np.zeros((support, support))
    k[o:o + 2, o:o + 2] = 0.25
    return k


def kernel_bicubic(support=6, a=-0.75):
    """Textbook bicubic kernel, no anti-aliasing (operator-shift probe)."""
    def w(t):
        t = abs(t)
        if t <= 1:
            return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
        if t < 2:
            return a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a
        return 0.0
    v = np.array([w(d) for d in (-1.5, -0.5, 0.5, 1.5)])
    v = v / v.sum()
    o = support // 2 - 1
    k = np.zeros((support, support))
    k[o - 1:o + 3, o - 1:o + 3] = np.outer(v, v)
    return k


# ============================================================ losses ========


def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def _gwin(ws, sigma, device, dtype):
    g = torch.arange(ws, device=device, dtype=dtype) - (ws - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.t() @ g).unsqueeze(0).unsqueeze(0)


def ssim(x, y, ws=11, sigma=1.5, L=1.0):
    w = _gwin(ws, sigma, x.device, x.dtype)
    p = ws // 2
    mx, my = F.conv2d(x, w, padding=p), F.conv2d(y, w, padding=p)
    mx2, my2, mxy = mx * mx, my * my, mx * my
    sx = F.conv2d(x * x, w, padding=p) - mx2
    sy = F.conv2d(y * y, w, padding=p) - my2
    sxy = F.conv2d(x * y, w, padding=p) - mxy
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    return (((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) * (sx + sy + C2))).mean()


def fft_l1(x, y):
    return (torch.abs(torch.fft.rfft2(x.float(), norm="ortho"))
            - torch.abs(torch.fft.rfft2(y.float(), norm="ortho"))).abs().mean()


class Criterion(nn.Module):
    """Every term is evaluated in fp32. Under fp16 autocast the FFT term
    overflows to inf, the GradScaler then skips every optimizer step, and
    validation PSNR prints an identical number forever - the frozen-metric
    failure mode that cost the team a full run."""

    def __init__(self, w_pix=1.0, w_ssim=0.15, w_fft=0.05, w_lpips=0.0, lpips_fn=None):
        super().__init__()
        self.w_pix, self.w_ssim = w_pix, w_ssim
        self.w_fft, self.w_lpips = w_fft, w_lpips
        self.lpips_fn = lpips_fn

    def forward(self, pred, gt, w_lpips=None):
        pred, gt = pred.float(), gt.float()
        wl = self.w_lpips if w_lpips is None else w_lpips
        parts = {}
        loss = self.w_pix * charbonnier(pred, gt)
        parts["pix"] = float(loss.detach())
        if self.w_ssim > 0:
            s = 1 - ssim(pred.clamp(0, 1), gt)
            loss = loss + self.w_ssim * s
            parts["ssim"] = float(s.detach())
        if self.w_fft > 0:
            f = fft_l1(pred, gt)
            loss = loss + self.w_fft * f
            parts["fft"] = float(f.detach())
        if wl > 0 and self.lpips_fn is not None:
            lp = self.lpips_fn(pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1,
                               gt.repeat(1, 3, 1, 1) * 2 - 1).mean()
            loss = loss + wl * lp
            parts["lpips"] = float(lp.detach())
        return loss, parts


def psnr_np(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


@torch.no_grad()
def ssim_np(a, b):
    """Wang et al. standard SSIM: 11x11 Gaussian window, sigma=1.5.

    This is the variant almost all restoration papers report, and the one our
    SSIM loss optimises. Note that scikit-image's DEFAULT arguments give a
    different quantity (7x7 uniform window, sample covariance) - both public
    reference submissions report that default without saying so, and one of
    them optimises 11x11 Gaussian while reporting 7x7 uniform. We report both,
    labelled, so the number can be compared either way.
    """
    return float(ssim(torch.from_numpy(a).float()[None, None],
                      torch.from_numpy(b).float()[None, None]))


def ssim_np_skimage(a, b, data_range=1.0):
    """scikit-image SSIM with DEFAULT arguments (7x7 uniform window)."""
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")
    return float(structural_similarity(b, np.clip(a, 0, 1), data_range=data_range))


# ========================================================== degrader ========


class Degrader:
    """Split-path degradation matched to the calibration, with domain
    randomisation on top.

    beta of the noise VARIANCE is injected before the resampler and (1-beta)
    after. Because the kernel attenuates upstream noise by sum(K^2), the
    per-path sigmas that preserve the measured LR-domain variance law are
        upstream:   sigma * sqrt(beta / sum(K^2))
        downstream: sigma * sqrt(1 - beta)

    Domain randomisation (this is the OOD lever): the held-out test set is
    stated to contain content from other sources with noise levels that "may
    vary within a similar range". Training on ONE exactly-calibrated operator
    invites overfitting to it, so sigma, beta and the kernel itself are
    jittered around the MEASURED values - randomising around a measurement
    rather than around a guess.
    """

    def __init__(self, K, sg, ss, beta, jitter=0.25, beta_jitter=0.0,
                 kernel_jitter=0.0, device="cpu"):
        self.K0 = torch.as_tensor(np.asarray(K), dtype=torch.float32)[None, None].to(device)
        self.sg, self.ss = float(sg), float(ss)
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.jitter = float(jitter)
        self.bj = float(beta_jitter)
        self.kj = float(kernel_jitter)
        self.support = self.K0.shape[-1]
        self.off = self.support // 2 - 1
        self.device = device

    def to(self, dev):
        self.K0 = self.K0.to(dev)
        self.device = dev
        return self

    def _down(self, x, K):
        n0, n1 = x.shape[-2] // 2, x.shape[-1] // 2
        p = self.support
        y = F.conv2d(F.pad(x, (p, p, p, p), mode="reflect"), K.to(x.dtype))
        s = p - self.off
        return y[:, :, s::2, s::2][:, :, :n0, :n1]

    def __call__(self, hr):
        B = hr.shape[0]
        dev = hr.device
        # --- kernel jitter (per batch: one conv) ---------------------------
        K = self.K0.to(dev)
        if self.kj > 0 and float(torch.rand(1)) < 0.5:
            K = K + torch.randn_like(K) * (self.kj * K.abs().mean())
            K = K / K.sum()
        sumk2 = float((K ** 2).sum())
        # --- beta jitter ---------------------------------------------------
        beta = self.beta
        if self.bj > 0:
            beta = float(np.clip(beta + np.random.uniform(-self.bj, self.bj), 0.0, 1.0))
        a_hr = math.sqrt(beta / max(sumk2, 1e-8))
        a_lr = math.sqrt(max(1.0 - beta, 0.0))
        # --- per-sample sigma jitter (log-uniform) -------------------------
        j = self.jitter
        u = torch.empty(B, 1, 1, 1, device=dev).uniform_(-j, j)
        ss = self.ss * torch.exp(u)
        u2 = torch.empty(B, 1, 1, 1, device=dev).uniform_(-j, j)
        sg = self.sg * torch.exp(u2)

        x = hr
        if beta > 1e-4:
            x = x * (1 + torch.randn_like(x) * (ss * a_hr)) \
                + torch.randn_like(x) * (sg * a_hr)
        lr = self._down(x, K)
        if a_lr > 1e-4:
            lr = lr * (1 + torch.randn_like(lr) * (ss * a_lr)) \
                 + torch.randn_like(lr) * (sg * a_lr)
        return lr


# ======================================================= calibration ========


def solve_kernel(pairs, support=6):
    """LS solve of LR[i,j] = sum_ab K[a,b] * GT[2i+a-off, 2j+b-off].

    E[LR | GT] = D(GT) holds for EVERY ordering of the three degradations,
    because speckle is mean-1 multiplicative, Gaussian noise is mean-0
    additive and D is linear. So the undisclosed order never has to be
    identified: D is recoverable by ordinary least squares regardless.
    """
    o = support // 2 - 1
    A_acc = np.zeros((support * support, support * support), np.float64)
    b_acc = np.zeros(support * support, np.float64)
    for gt, lr in pairs:
        n0, n1 = lr.shape
        gtp = np.pad(gt, support, mode="reflect")
        cols = []
        for a in range(support):
            for b in range(support):
                cols.append(gtp[support + a - o::2, support + b - o::2][:n0, :n1].reshape(-1))
        A = np.stack(cols, 1).astype(np.float64)
        y = lr.reshape(-1).astype(np.float64)
        A_acc += A.T @ A
        b_acc += A.T @ y
    K = np.linalg.solve(A_acc + 1e-8 * np.eye(support * support), b_acc)
    return K.reshape(support, support)


def apply_kernel(gt, K):
    s = K.shape[0]
    o = s // 2 - 1
    n0, n1 = gt.shape[0] // 2, gt.shape[1] // 2
    gtp = np.pad(gt, s, mode="reflect")
    out = np.zeros((n0, n1))
    for a in range(s):
        for b in range(s):
            out += K[a, b] * gtp[s + a - o::2, s + b - o::2][:n0, :n1]
    return out


def fit_noise(pairs, K, nbins=40):
    """Regress residual variance on intensity: Var = sigma_g^2 + sigma_s^2 mu^2."""
    mus, res = [], []
    for gt, lr in pairs:
        mu = apply_kernel(gt, K)
        mus.append(mu.reshape(-1))
        res.append((lr - mu).reshape(-1))
    mu = np.concatenate(mus)
    r = np.concatenate(res)
    lo, hi = np.percentile(mu, [0.5, 99.5])
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(mu, edges) - 1, 0, nbins - 1)
    mc, vc, wc = [], [], []
    for k in range(nbins):
        m = idx == k
        if m.sum() < 500:
            continue
        mc.append(mu[m].mean())
        vc.append(r[m].var())
        wc.append(m.sum())
    if len(mc) < 3:
        v = float(r.var())
        return dict(sigma_g=math.sqrt(max(v, 0)), sigma_s=1e-3, r2=0.0,
                    mc=np.array([0.0]), vc=np.array([v]), pred=np.array([v]))
    mc, vc = np.array(mc), np.array(vc)
    wc = np.array(wc, np.float64)
    X = np.stack([np.ones_like(mc), mc ** 2], 1)
    W = np.diag(wc / wc.sum())
    a, b = np.linalg.solve(X.T @ W @ X, X.T @ W @ vc)
    pred = X @ np.array([a, b])
    denom = (wc * (vc - np.average(vc, weights=wc)) ** 2).sum()
    r2 = 1 - (wc * (vc - pred) ** 2).sum() / max(denom, 1e-12)
    return dict(sigma_g=float(np.sqrt(max(a, 0))), sigma_s=float(np.sqrt(max(b, 0))),
                r2=float(r2), mc=mc, vc=vc, pred=pred)


def autocorr(pairs, K):
    hs, vs = [], []
    for gt, lr in pairs:
        r = lr - apply_kernel(gt, K)
        r = r - r.mean()
        v = r.var()
        if v <= 0:
            continue
        hs.append((r[:, :-1] * r[:, 1:]).mean() / v)
        vs.append((r[:-1, :] * r[1:, :]).mean() / v)
    if not hs:
        return 0.0, 0.0
    return float(np.mean(hs)), float(np.mean(vs))


# ================================================================ io ========

IMG_EXT = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")


def list_inputs(d):
    """Every restorable file in a directory, .npy first then images."""
    files = sorted(glob.glob(os.path.join(d, "*.npy")))
    for e in IMG_EXT:
        files += sorted(glob.glob(os.path.join(d, "*" + e)))
        files += sorted(glob.glob(os.path.join(d, "*" + e.upper())))
    return sorted(set(files))


def load_any(path):
    """Load .npy or an image as float32. Returns (array, meta).

    NoisyLR .npy files are already float and may fall outside [0,1]; that is
    intentional and is preserved exactly. Integer images are scaled by their
    dtype maximum.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path)
        return np.asarray(a, dtype=np.float32), dict(kind="npy", dtype=str(a.dtype))
    from PIL import Image
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim == 3:
        a = a.mean(-1)
    info = dict(kind="img", dtype=str(a.dtype))
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
        info["max"] = 255
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
        info["max"] = 65535
    else:
        a = a.astype(np.float32)
        info["max"] = 1
    return a, info


def save_png16(path, arr):
    """16-bit PNG so the visual copy keeps ~1/65535 precision."""
    from PIL import Image
    a = np.clip(arr, 0.0, 1.0)
    q = np.round(a * 65535.0).astype(np.uint16)
    try:
        Image.fromarray(q, mode="I;16").save(path)
    except Exception:
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def save_outputs(out_dir, stem, arr, save_npy=True, save_png=True, png_dir=None):
    """Write the restored image as .npy (scored artifact) and .png (visual).

    KLA scores the images exactly as saved and does not clip or renormalise,
    so clipping to [0,1] happens HERE. GT lives in [0,1] and clipping is a
    projection onto a convex set containing the target, so it can only reduce
    error.
    """
    paths = []
    a = np.asarray(arr, dtype=np.float32)
    if save_npy:
        p = os.path.join(out_dir, stem + ".npy")
        np.save(p, a)
        paths.append(p)
    if save_png:
        d = png_dir if png_dir else out_dir
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, stem + ".png")
        save_png16(p, a)
        paths.append(p)
    return paths


# ==================================================== self-ensemble =========


@torch.no_grad()
def forward_se(model, x, n=1):
    """Geometric self-ensemble over the dihedral group. Averaging in the MMSE
    sense can only help PSNR; the cost is n forward passes, so n is the
    quality/throughput dial."""
    if n <= 1:
        return model(x)
    tfs = [lambda t: t, lambda t: t.flip(-1), lambda t: t.flip(-2),
           lambda t: t.flip(-1).flip(-2)]
    inv = [lambda t: t, lambda t: t.flip(-1), lambda t: t.flip(-2),
           lambda t: t.flip(-1).flip(-2)]
    if n >= 8:
        tfs += [lambda t: t.transpose(-1, -2),
                lambda t: t.transpose(-1, -2).flip(-1),
                lambda t: t.transpose(-1, -2).flip(-2),
                lambda t: t.transpose(-1, -2).flip(-1).flip(-2)]
        inv += [lambda t: t.transpose(-1, -2),
                lambda t: t.flip(-1).transpose(-1, -2),
                lambda t: t.flip(-2).transpose(-1, -2),
                lambda t: t.flip(-1).flip(-2).transpose(-1, -2)]
    outs = [g(model(f(x))) for f, g in zip(tfs[:n], inv[:n])]
    return torch.stack(outs).mean(0)


def pad_to_even(x, mult=2):
    """PixelShuffle x2 needs even spatial dims; pad by reflection and record
    the crop so ANY input size works without the evaluator editing anything."""
    h, w = x.shape[-2], x.shape[-1]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def raw(m):
    """DataParallel prefixes every state_dict key with 'module.'."""
    return m.module if isinstance(m, nn.DataParallel) else m


# ===================================================================== CLI ===
# Standalone inference for the KLA restoration challenge.
#
#     python inference.py --input_dir <dir> --output_dir <dir>
#
# Reads every .npy (or .png/.tif) in input_dir, restores it, and writes a
# float32 .npy of the same name to output_dir plus a 16-bit .png preview.
# Output is clipped to [0,1]: GT lives in [0,1] and KLA does not clip, and
# clipping is a projection onto a convex set containing the target, so it can
# only reduce error.
import argparse, time, json
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser(description="KLA PS-1 restoration inference")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--weights", default=os.path.join(here, "weights", "model.pt"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--self_ensemble", type=int, default=None,
                    help="1 = fastest, 4 = flips, 8 = flips+transpose")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 inference")
    ap.add_argument("--no_png", action="store_true", help="write .npy only")
    ap.add_argument("--png_subdir", default="png",
                    help="subfolder for .png previews ('' = alongside the .npy)")
    a = ap.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    png_dir = os.path.join(a.output_dir, a.png_subdir) if a.png_subdir else a.output_dir
    if not a.no_png:
        os.makedirs(png_dir, exist_ok=True)

    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    cfg, cal = ck["cfg"], ck["calib"]
    calib = dict(sigma_g=cal["sigma_g"], sigma_s=cal["sigma_s"], K=np.array(cal["K"]))
    net = build_net(cfg, calib, a.device)
    net.load_state_dict(ck["model"])
    net.eval()
    se = a.self_ensemble if a.self_ensemble is not None else ck.get("self_ensemble", 1)
    half = (a.device == "cuda") and not a.fp32
    if half:
        net = net.half()
    try:
        net = net.to(memory_format=torch.channels_last)
    except Exception:
        pass

    files = list_inputs(a.input_dir)
    print(f"{len(files)} files | device={a.device} | batch={a.batch_size} | SE={se} "
          f"| fp16={half} | png={not a.no_png}")
    if not files:
        print("nothing to do"); return

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.workers) as ex, torch.no_grad():
        for b0 in range(0, len(files), a.batch_size):
            chunk = files[b0:b0 + a.batch_size]
            loaded = list(ex.map(load_any, chunk))
            # group by shape so mixed 128x128 / 256x256 inputs both batch cleanly
            groups = {}
            for pth, (arr, meta) in zip(chunk, loaded):
                groups.setdefault(arr.shape, []).append((pth, arr))
            for shape, items in groups.items():
                x = torch.from_numpy(np.stack([a_ for _, a_ in items]))[:, None]
                x, h0, w0 = pad_to_even(x)
                x = x.to(a.device, non_blocking=True)
                if half:
                    x = x.half()
                try:
                    x = x.contiguous(memory_format=torch.channels_last)
                except Exception:
                    pass
                y = forward_se(net, x, se)
                y = y.float()[:, :, :2 * h0, :2 * w0].clamp(0, 1).cpu().numpy()[:, 0]
                for (pth, _), arr_out in zip(items, y):
                    stem = os.path.splitext(os.path.basename(pth))[0]
                    save_outputs(a.output_dir, stem, arr_out.astype(np.float32),
                                 save_npy=True, save_png=not a.no_png,
                                 png_dir=png_dir)
    if a.device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"done: {dt:.3f}s total | {dt/max(1,len(files))*1e3:.3f} ms/img | "
          f"{len(files)/dt:.1f} img/s")


if __name__ == "__main__":
    main()
