#!/usr/bin/env python3
"""Reproduce the submitted checkpoint from scratch.

    python train.py --data_dir /path/to/train --out_dir runs/repro

Expects <data_dir>/GT/*.npy and <data_dir>/NoisyLR/*.npy. Every stage the
notebook performs - calibration, the GPU-resident sampler, the wall-clock-aware
cosine schedule, EMA, the divergence guard - is reproduced here; the notebook is
the annotated version of this file.
"""
import os, sys, json, time, math, glob, copy, random, argparse
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from kla_core import *  # noqa: F401,F403


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="runs/repro")
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "config.json"))
    ap.add_argument("--budget_min", type=float, default=None,
                    help="override total training minutes (default: from config.json)")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    if a.budget_min is not None:
        cfg["total_budget_min"] = a.budget_min
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg.get("seed", 1337)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gt_dir = os.path.join(a.data_dir, "GT")
    lr_dir = os.path.join(a.data_dir, "NoisyLR")
    names = sorted(set(os.path.basename(p) for p in glob.glob(gt_dir + "/*.npy")) &
                   set(os.path.basename(p) for p in glob.glob(lr_dir + "/*.npy")))
    assert names, f"no paired .npy under {a.data_dir}"
    L = lambda d, n: np.load(os.path.join(d, n)).astype(np.float32)

    # ---- stage 0: calibration ---------------------------------------------
    cal = [(L(gt_dir, n), L(lr_dir, n)) for n in names[:min(120, len(names))]]
    K = solve_kernel(cal, 6)
    nz = fit_noise(cal, K)
    ac = float(np.mean(autocorr(cal, K)))
    ac_hr = float((K[:-2, :] * K[2:, :]).sum() / (K ** 2).sum())
    beta = float(np.clip(ac / ac_hr, 0, 1)) if abs(ac_hr) > 1e-6 else 0.0
    calib = dict(K=K.tolist(), sigma_g=nz["sigma_g"], sigma_s=nz["sigma_s"],
                 beta=beta, r2=nz["r2"])
    json.dump(calib, open(os.path.join(a.out_dir, "calibration.json"), "w"), indent=2)
    print(f"calibrated: sg={nz['sigma_g']:.5f} ss={nz['sigma_s']:.5f} beta={beta:.3f}")

    # ---- data --------------------------------------------------------------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    nv = max(8, int(len(names) * cfg.get("val_frac", 0.08)))
    vn = [names[i] for i in perm[:nv]]; tn = [names[i] for i in perm[nv:]]
    json.dump({"train": tn, "val": vn}, open(os.path.join(a.out_dir, "split.json"), "w"))
    store = torch.float16 if dev == "cuda" else torch.float32
    LRT = torch.from_numpy(np.stack([L(lr_dir, n) for n in tn])).to(dev, dtype=store)
    GTT = torch.from_numpy(np.stack([L(gt_dir, n) for n in tn])).to(dev, dtype=store)
    VGT = [L(gt_dir, n) for n in vn]; VLR = [L(lr_dir, n) for n in vn]
    deg = Degrader(K, nz["sigma_g"], nz["sigma_s"], beta,
                   jitter=cfg.get("jitter_train", 0.3),
                   beta_jitter=cfg.get("beta_jitter", 0.12),
                   kernel_jitter=cfg.get("kernel_jitter", 0.05), device=dev)

    def gather(T, idx, i, j, s):
        ar = torch.arange(s, device=T.device)
        return T[idx[:, None, None], (i[:, None] + ar[None, :])[:, :, None],
                 (j[:, None] + ar[None, :])[:, None, :]]

    H = LRT.shape[-1]
    def batch(bs, P, sr):
        ns = int(round(bs * sr)); nr = bs - ns
        lr = gt = None
        if nr:
            k = torch.randint(0, LRT.shape[0], (nr,), device=dev)
            i = torch.randint(0, H - P + 1, (nr,), device=dev)
            j = torch.randint(0, H - P + 1, (nr,), device=dev)
            lr = gather(LRT, k, i, j, P)[:, None].float()
            gt = gather(GTT, k, 2 * i, 2 * j, 2 * P)[:, None].float()
        if ns:
            Hh = GTT.shape[-1]
            k = torch.randint(0, GTT.shape[0], (ns,), device=dev)
            i = torch.randint(0, (Hh - 2 * P) // 2 + 1, (ns,), device=dev) * 2
            j = torch.randint(0, (Hh - 2 * P) // 2 + 1, (ns,), device=dev) * 2
            hr = gather(GTT, k, i, j, 2 * P)[:, None].float()
            with torch.no_grad():
                s = deg(hr)
            lr = s if lr is None else torch.cat([lr, s])
            gt = hr if gt is None else torch.cat([gt, hr])
        if np.random.rand() < .5: lr, gt = lr.flip(-1), gt.flip(-1)
        if np.random.rand() < .5: lr, gt = lr.flip(-2), gt.flip(-2)
        r = int(np.random.randint(4))
        if r: lr, gt = torch.rot90(lr, r, (-2, -1)), torch.rot90(gt, r, (-2, -1))
        return lr.contiguous(), gt.contiguous()

    # ---- model / opt -------------------------------------------------------
    net = build_net(cfg, dict(sigma_g=nz["sigma_g"], sigma_s=nz["sigma_s"], K=K), dev)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.3f}M")
    bs = cfg.get("batch_size", 32)
    if cfg.get("multi_gpu", True) and torch.cuda.device_count() > 1:
        bs *= torch.cuda.device_count(); net = nn.DataParallel(net)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr_init"], weight_decay=1e-5,
                            betas=(0.9, 0.99))
    use_scaler = dev == "cuda" and torch.cuda.get_device_capability(0)[0] < 8
    amp_dt = torch.float16 if use_scaler else (
        torch.bfloat16 if dev == "cuda" else torch.float32)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    crit = Criterion(cfg["w_pix"], cfg["w_ssim"], cfg["w_fft"], 0.0, None)
    shadow = {k: v.detach().clone().float() for k, v in raw(net).state_dict().items()}
    budget = cfg.get("total_budget_min", cfg.get("time_budget_min", 300)) * 60
    t0 = time.time(); best = -1e9
    P = cfg["lr_patch"]

    for step in range(cfg["total_steps"] + 1):
        el = time.time() - t0
        frac = min(1.0, max((step - cfg["warmup"]) / max(1, cfg["total_steps"] - cfg["warmup"]),
                            el / budget))
        if frac >= cfg.get("patch_grow_at", 0.7):
            P = cfg.get("lr_patch_late", cfg["lr_patch"])
        lr = (cfg["lr_init"] * step / max(1, cfg["warmup"])) if step < cfg["warmup"] else              cfg["lr_min"] + 0.5 * (cfg["lr_init"] - cfg["lr_min"]) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(bs, P, cfg["synth_ratio"])
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=amp_dt) if dev == "cuda" else             __import__("contextlib").nullcontext()
        with ctx:
            out = net(x)
        loss, _ = crit(out, y)
        assert torch.isfinite(loss), f"non-finite loss at {step}"
        if use_scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for k, v in raw(net).state_dict().items():
                if v.dtype.is_floating_point:
                    shadow[k].mul_(cfg["ema_decay"]).add_(v.detach().float(),
                                                          alpha=1 - cfg["ema_decay"])
                else:
                    shadow[k] = v.detach().clone().float()
        if step % 500 == 0:
            print(f"{step:7d}  loss {float(loss):.5f}  lr {lr:.2e}  "
                  f"{step/max(1e-9, el):.2f} it/s")
        if step and step % cfg["val_every"] == 0:
            bak = {k: v.detach().clone() for k, v in raw(net).state_dict().items()}
            cur = raw(net).state_dict()
            raw(net).load_state_dict({k: shadow[k].to(dtype=cur[k].dtype) for k in cur})
            m = raw(net); m.eval()
            ps = []
            with torch.no_grad():
                for i in range(min(64, len(VGT))):
                    xt = torch.from_numpy(VLR[i])[None, None].to(dev)
                    o = m(xt).float().clamp(0, 1)[0, 0].cpu().numpy()
                    ps.append(psnr_np(o, VGT[i]))
            m.train()
            p = float(np.mean(ps))
            print(f"   VAL {step}: PSNR {p:.3f}")
            if p > best:
                best = p
                torch.save(dict(model=raw(net).state_dict(), calib=calib, cfg=cfg,
                                step=step, metrics=dict(psnr=p)),
                           os.path.join(a.out_dir, "model.pt"))
            raw(net).load_state_dict(bak)
        if el > budget:
            print(f"budget reached at step {step}"); break
    print(f"done. best val PSNR {best:.3f} dB -> {a.out_dir}/model.pt")


if __name__ == "__main__":
    main()
