#!/usr/bin/env python3
"""Architecture of DeepDAS Ensemble v1 at inference with the explicit tailgating summary feature NEUTRALIZED (vector figure).

Drawn from the implementation, not from the illustrative references: every shape, kernel, stride, layer width, activation, the parameter
count, the number of seeds and tau are read from the frozen checkpoints / ensemble.json of this release copy and asserted.
The trained head is 17->16->1. In the neutralized option the 17th input (standardised phi_u) is held at 0 at inference; the figure
shows it as a hatched cell. There is no max-pooling and no dropout in the network: the only pooling is mean/std over the window codes.
Input traces: one real 64-bin window of a normal training-region track (fixed selection rule; recorded in the spec file).
Output: ../figures/{pdf,png}/G0_architecture_phi_neutralized.* and ../figures/data/G0_architecture_spec.json
"""
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
os.environ["DEEPDAS_PHI_MODE"] = "neutralized"
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, Circle  # noqa: E402
from deepdas.ensemble.detector import load_detector  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
for d in ("pdf", "png", "data"):
    (FIG / d).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7,
                     "pdf.fonttype": 42, "savefig.dpi": 300, "hatch.linewidth": 0.5})
TXT_W = 6.93

# ---------------------------------------------------------------- verify against the frozen model (all five seeds)
SPEC = json.loads((ROOT / "models/deepdas_ensemble_v1/ensemble.json").read_text()); TAU = SPEC["decision"]["threshold"]
dets = [load_detector(ROOT / "models/deepdas_ensemble_v1" / s["checkpoint"], 1) for s in SPEC["seeds"]]
d0 = dets[0]; h = torch.zeros(2, 14, 64); SH = [(14, 64)]; KS = []; ACT = set()
for m in d0.enc:
    if isinstance(m, torch.nn.Conv1d):
        KS.append((m.kernel_size[0], m.stride[0])); h = m(h); SH.append((h.shape[1], h.shape[2]))
    elif isinstance(m, torch.nn.GELU):
        h = m(h); ACT.add("GELU")
    else:
        raise AssertionError(f"unexpected layer {m}")               # no max-pooling, no dropout
FLAT = h.flatten(1).shape[1]; ZD = d0.to_z.out_features
HEAD = [d0.head[0].in_features, d0.head[0].out_features, d0.head[2].out_features]; NPAR = sum(p.numel() for p in d0.parameters())
assert SH == [(14, 64), (32, 64), (64, 32), (64, 16), (128, 8)] and KS == [(7, 1), (5, 2), (5, 2), (3, 2)] and ACT == {"GELU"}
assert FLAT == 1024 and ZD == 8 and HEAD == [17, 16, 1] and NPAR == 67225 and len(dets) == 5
assert all(d.phi_mode == "neutralized" and d.K == 1 for d in dets)
# the neutralized head input is exactly 0 for the 17th coordinate, whatever phi_u is
f = torch.randn(3, 16); e = torch.tensor([[0.0], [0.5], [0.81]])
assert torch.equal(d0.head_in(f, e)[:, 16], torch.zeros(3))


def real_window():
    sp = json.loads((ROOT / "data/processed/normal_corpus/splits.json").read_text()); train = set(sp["train"])
    c = pd.read_parquet(ROOT / "data/processed/normal_corpus/channels.parquet", columns=["track_key", "bin", "v_lon", "a_lon", "d_lat", "th_lead", "m_lead", "m_valid"])
    a = pd.read_parquet(ROOT / "data/processed/normal_corpus/amplitude.parquet", columns=["track_key", "bin", "amp_log10"]).set_index(["track_key", "bin"]).amp_log10
    c = c[c.track_key.map(lambda k: k.split("/")[0] in train and "/normal" in k)]
    for key, g in c.groupby("track_key", sort=True):
        g = g.sort_values("bin").reset_index(drop=True)
        for s0 in range(10, len(g) - 64, 8):
            w = g.iloc[s0:s0 + 64]
            if not ((w.m_valid > 0).all() and (w.m_lead > 0.5).all() and (w.th_lead < 5).all() and w.th_lead.std() > 0.15 and w.d_lat.std() > 0.05):
                continue
            amp = a.reindex(list(zip([key] * 64, w.bin))).to_numpy()
            if np.isfinite(amp).mean() >= 0.85:
                return key, (int(w.bin.iloc[0]), int(w.bin.iloc[-1])), dict(amp=amp, v_lon=w.v_lon.to_numpy(), a_lon=w.a_lon.to_numpy(), th=w.th_lead.to_numpy())
    raise RuntimeError("no window")


BLUE = ("#D6E2F3", "#1F3F73"); GREY = ("#E9E9E9", "#444444"); GREEN = ("#DCECDF", "#245A38"); INK = "#111111"


def stack(ax, x, yc, L, C, col, unit=0.0145, w=0.07, dx=0.016, dy=0.016):
    """Staggered stack of rectangles: height = L*unit (time bins), number of rectangles = max(2, C/8) (channels)."""
    n = int(max(2, round(C / 8))); hg = L * unit; y0 = yc - (hg + (n - 1) * dy) / 2
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + i * dx, y0 + i * dy), w, hg, fc=col[0], ec=col[1], lw=0.4, zorder=10 + (n - i) * 0.01))
    return dict(x=x, right=x + w + (n - 1) * dx, top=y0 + hg + (n - 1) * dy, bottom=y0, cx=x + (w + (n - 1) * dx) / 2)


def vec(ax, x, yc, n, cells, w=0.07, cell=0.05):
    y0 = yc - n * cell / 2; rects = []
    for k in range(n):
        spec = cells[k] if isinstance(cells, list) else cells
        r = Rectangle((x, y0 + k * cell), w, cell, fc=spec.get("fc"), ec=spec.get("ec"), lw=0.4, hatch=spec.get("hatch"), zorder=10); ax.add_patch(r); rects.append(r)
    return dict(x=x, right=x + w, y0=y0, top=y0 + n * cell, cx=x + w / 2, cell=cell)


def arrow(ax, x0, y0, x1, y1, lw=0.7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=lw, color=INK, mutation_scale=7, shrinkA=0, shrinkB=0), zorder=30)


def main():
    H = 2.15; fig = plt.figure(figsize=(TXT_W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, TXT_W); ax.set_ylim(0, H); ax.axis("off")
    YC, YT, YB = 1.12, 1.98, 0.36           # centre line, layer-name row, parameter row
    key, bins, W = real_window()
    # input window (real)
    ox, ow, oy0, oy1 = 0.30, 0.46, 0.66, 1.62
    ax.add_patch(Rectangle((ox, oy0), ow, oy1 - oy0, fill=False, ec="#444444", lw=0.5))
    for i, (k, lab) in enumerate((("amp", "DAS"), ("v_lon", r"$v$"), ("a_lon", r"$a$"), ("th", r"$t_h$"))):
        s = np.asarray(W[k], float); s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-9); yc = oy1 - 0.14 - i * 0.215
        ax.plot(ox + 0.02 + np.arange(64) / 63 * (ow - 0.04), yc - 0.07 + 0.14 * s, color="#2F5D9A", lw=0.55)
        ax.text(ox - 0.03, yc, lab, ha="right", va="center", fontsize=6.2)
    ax.text(ox + ow / 2, oy0 + 0.05, r"$\vdots$  (14 ch.)", ha="center", va="bottom", fontsize=5.8, color="#555555")
    ax.text(ox + ow / 2, YT, "Input window", ha="center", fontsize=7, fontweight="bold")
    ax.text(ox + ow / 2, YB + 0.12, "6.4 s, 10 Hz", ha="center", fontsize=6.4)
    # encoder
    xs = [0.93, 1.32, 1.78, 2.17, 2.55]; st = []
    for (C, L), xx, col in zip(SH, xs, [GREY] + [BLUE] * 4):
        st.append(stack(ax, xx, YC, L, C, col))
    arrow(ax, ox + ow + 0.02, YC, st[0]["x"] - 0.03, YC)
    for a_, b_ in zip(st[:-1], st[1:]):
        arrow(ax, a_["right"] + 0.03, YC, b_["x"] - 0.03, YC)
    names = ["Input tensor", "Conv1d + GELU", "Conv1d + GELU", "Conv1d + GELU", "Conv1d + GELU"]
    for s_, (C, L), nm, ks in zip(st, SH, names, [None] + KS):
        ax.text(s_["cx"], YT, nm.replace(" + ", "\n+ ").replace("Input tensor", "Input\ntensor"), ha="center", va="center", fontsize=6.6, fontweight="bold", linespacing=1.0)
        ax.text(s_["cx"], YB + 0.12, f"{C}×{L}", ha="center", fontsize=6.6)
        if ks:
            ax.text(s_["cx"], YB - 0.02, f"$k$={ks[0]}, $s$={ks[1]}", ha="center", fontsize=6.0)
    ax.texts[[t.get_text() for t in ax.texts].index("Input window")].set_visible(True)
    # flatten + linear
    fl = vec(ax, 3.02, YC, 1, {"fc": BLUE[0], "ec": BLUE[1]}, w=0.045, cell=0.92)
    arrow(ax, st[-1]["right"] + 0.03, YC, fl["x"] - 0.03, YC)
    ax.text(fl["cx"], YT, "Flatten", ha="center", fontsize=6.6, fontweight="bold"); ax.text(fl["cx"], YB + 0.12, f"{FLAT}", ha="center", fontsize=6.6)
    z = vec(ax, 3.36, YC, ZD, {"fc": BLUE[0], "ec": BLUE[1]}, w=0.06, cell=0.06)
    arrow(ax, fl["right"] + 0.03, YC, z["x"] - 0.03, YC)
    ax.text(z["cx"], YT, "Linear", ha="center", fontsize=6.6, fontweight="bold"); ax.text(z["cx"], YB + 0.12, r"$\mathbf{z}_w\in\mathbb{R}^{8}$", ha="center", fontsize=6.6)
    ax.text(z["cx"], YB - 0.02, f"{FLAT}→{ZD}", ha="center", fontsize=6.0)
    # pooling over the W windows of the track
    px = 3.76
    for i in range(4, -1, -1):
        vec(ax, px + i * 0.018, YC + i * 0.018 - 0.03, ZD, {"fc": BLUE[0], "ec": BLUE[1]}, w=0.045, cell=0.045)
    arrow(ax, z["right"] + 0.03, YC, px - 0.03, YC)
    ax.text(px + 0.06, YT, "Mean ⊕ std\n(windows)", ha="center", va="center", fontsize=6.6, fontweight="bold", linespacing=1.0)
    ax.text(px + 0.06, YB + 0.12, r"$\mathbf{z}_1..\mathbf{z}_W$", ha="center", fontsize=6.4)
    # head input 17: 16 learned (varying) + 1 held at 0
    cells = [{"fc": "white", "ec": "#777777", "hatch": "////////"}] + [{"fc": BLUE[0], "ec": BLUE[1]}] * 16
    cat = vec(ax, 4.27, YC, HEAD[0], cells, w=0.06, cell=0.052)
    arrow(ax, px + 0.15, YC + 0.1, cat["x"] - 0.03, YC + 0.1)
    ax.text(cat["cx"], YT, "Head\ninput", ha="center", va="center", fontsize=6.6, fontweight="bold", linespacing=1.0)
    ax.text(cat["cx"], YB + 0.12, f"16 + 1 = {HEAD[0]}", ha="center", fontsize=6.6)
    ax.text(cat["right"] + 0.03, cat["y0"] + 0.026, "0", ha="left", va="center", fontsize=6.4)
    ax.text(cat["cx"], YB - 0.02, "hatched: 17th input\n(standardised $\\phi_u$) held at 0\nat inference (neutralized)", ha="center", va="top",
            fontsize=5.8, color="#444444", linespacing=1.05)
    ax.text(cat["x"] - 0.03, cat["y0"] + 0.052 * 9, r"$\boldsymbol{\mu}_u,\boldsymbol{\sigma}_u$", ha="right", va="center", fontsize=6.2)
    # MLP 17 -> 16 -> 1 (trained head of the checkpoint)
    hx = 4.78; hy = np.linspace(cat["top"] - 0.1, cat["y0"] + 0.1, 7); show = [0, 1, 2, 4, 5, 6]; oc = (5.1, YC)
    for k in range(17):
        yk = cat["y0"] + (k + 0.5) * cat["cell"]
        for i in show:
            ax.plot([cat["right"], hx - 0.035], [yk, hy[i]], color="#B5B5B5" if k else "#DDDDDD", lw=0.15, zorder=4)
    for i in show:
        ax.add_patch(Circle((hx, hy[i]), 0.035, fc=GREEN[0], ec=GREEN[1], lw=0.5, zorder=10)); ax.plot([hx + 0.035, oc[0] - 0.04], [hy[i], oc[1]], color="#8FA897", lw=0.3, zorder=4)
    ax.text(hx, hy[3], r"$\vdots$", ha="center", va="center", fontsize=7)
    ax.add_patch(Circle(oc, 0.04, fc=GREEN[0], ec=GREEN[1], lw=0.7, zorder=10))
    ax.text((cat["right"] + oc[0]) / 2 + 0.05, YT, f"MLP\n{HEAD[0]}→{HEAD[1]}→{HEAD[2]}", ha="center", va="center", fontsize=6.6, fontweight="bold", linespacing=1.0)
    ax.text(hx, YB + 0.12, "16, GELU", ha="center", fontsize=6.4); ax.text(oc[0], YB + 0.12, r"$\ell_{u,k}$", ha="center", fontsize=6.8)
    # ensemble
    ex = 5.42; ys = YC + np.linspace(0.42, -0.42, 5)
    for k, yy in enumerate(ys):
        ax.add_patch(Rectangle((ex, yy - 0.055), 0.27, 0.11, fc="white", ec=INK, lw=0.45 if k else 0.8, zorder=10))
        ax.text(ex + 0.135, yy, f"net {k + 1}", ha="center", va="center", fontsize=6.0, zorder=11)
        arrow(ax, ex + 0.28, yy, 5.98, YC + (yy - YC) * 0.15, lw=0.45)
    arrow(ax, oc[0] + 0.045, oc[1], ex - 0.02, ys[0], lw=0.5)
    ax.add_patch(Circle((6.06, YC), 0.075, fc="white", ec=INK, lw=0.7, zorder=10)); ax.text(6.06, YC, r"$s_u$", ha="center", va="center", fontsize=6.8, zorder=11)
    arrow(ax, 6.14, YC, 6.36, YC)
    ax.text(6.39, YC, "ALARM\nif $s_u>\\tau$", ha="left", va="center", fontsize=6.4, linespacing=1.05)
    ax.text(6.08, YT, f"Ensemble of {len(dets)}\ntrained networks", ha="center", va="center", fontsize=6.6, fontweight="bold", linespacing=1.0)
    ax.text(6.08, YB + 0.12, r"$s_u=\frac{1}{5}\sum_k \frac{\ell_{u,k}-\mu_k}{\sigma_k}$", ha="center", fontsize=6.4)
    ax.text(6.08, YB - 0.04, rf"$\tau$ = {TAU:.3f} (frozen)", ha="center", fontsize=6.0)
    fig.savefig(FIG / "pdf/G0_architecture_phi_neutralized.pdf", pad_inches=0.0); fig.savefig(FIG / "png/G0_architecture_phi_neutralized.png", pad_inches=0.0); plt.close(fig)
    json.dump(dict(phi_mode="neutralized (17th head input held at 0 in the standardised coordinate)", shapes=SH, kernels_strides=KS, activation="GELU",
                   pooling="mean and std of the window codes (no max-pooling, no dropout)", flatten=FLAT, latent=ZD, head=HEAD,
                   params_per_seed=NPAR, seeds=len(dets), tau=TAU, input_window_track=key, input_window_bins=bins), open(FIG / "data/G0_architecture_spec.json", "w"), indent=1)
    print("saved G0; window", key, bins)


if __name__ == "__main__":
    main()
