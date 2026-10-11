#!/usr/bin/env python3
"""
W1 / W2 training (weak_types_W1W2_20261010/plan/PLAN_W1_W2_FROZEN.md) = a copy of ../../stall_diversity_20261008/x_train_eval_s.py (the E_DIV recipe,
run there with STALLARM=diversity NEWDATA=1 clean=1 feats=tg) with ONLY these changes (diff in plan/DIFF_x_train_eval_w_vs_s.txt):
  * paths (this script lives two levels deeper);
  * the W batch (data/W_vehicles.csv): train anomalies / unaffected backgrounds / targeted controls -> train; all W val vehicles -> val;
    the 18 stallval stalls -> val (checkpoint AP on the expanded validation, plan section 6);
  * ARM env: W1 = the unchanged sampler (48 normals + 16 positives, uniform); W2 = 36 normals (original + new backgrounds) + 4 new targeted controls
    + 16 positives from the 105 original positives + 8 from the new positives (batch 64 in both arms);
  * phase 3 removed: every model (E_DIV, W1, W2) is scored by the same frozen scripts/w07_evaluate.py.
One run per call:  python x_train_eval_w.py <run_name> 1 tg <out_dir>   (env ARM=W1|W2, SEED, NTHREADS)
"""
import json, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score

RUN, CLEAN, FEATS, OUT = sys.argv[1], sys.argv[2] == "1", [f for f in sys.argv[3].split(",") if f and f != "none"], Path(sys.argv[4]).resolve()
HERE = Path(__file__).resolve().parent; DL = HERE.parents[2]; WDIR = HERE.parent
AMPX = DL / "experiments_followup/amplitude_only_20260927_203146"; CX = DL / "experiments_followup/supervised_crossregion_20260927_200203"
sys.path.insert(0, str(DL / "src")); sys.path.insert(0, str(AMPX / "scripts"))
from deepdas.data.channels import CHANNELS  # noqa: E402
from deepdas.data.dataset import window_starts  # noqa: E402
import amp_lib as L  # noqa: E402

(OUT / "model").mkdir(parents=True, exist_ok=True); (OUT / "evaluation").mkdir(exist_ok=True)
EVAL_ONLY = os.environ.get("EVAL_ONLY") == "1"          # re-score only: never trains, never touches checkpoint or thresholds
CKPT = Path(os.environ.get("CKPT", OUT / "model/checkpoint_best.pt")); THRF = Path(os.environ.get("THR", OUT / "model/FROZEN_THRESHOLDS.json"))
if (OUT / "model/done.json").exists() and not EVAL_ONLY:
    sys.exit("already trained - refusing to overwrite")
if EVAL_ONLY and not (CKPT.exists() and THRF.exists()):
    sys.exit("EVAL_ONLY needs an existing checkpoint and frozen thresholds")
SEED = int(os.environ.get("SEED", "0")); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True); torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
T, STRIDE, MAXW, NB, PB, STEPS, EPOCHS, PATIENCE, BIN = 64, 8, 32, 48, 16, 120, 25, 6, 0.1
CI = {c: i for i, c in enumerate(CHANNELS)}
log = open(OUT / ("model/eval_log.txt" if os.environ.get("EVAL_ONLY") == "1" else "model/train_log.txt"), "a")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


say(json.dumps({"run": RUN, "newdata": os.environ.get("NEWDATA") == "1", "seed": SEED, "clean": CLEAN, "feats": FEATS, "torch": torch.__version__}))


# ------------------------------------------------------------------ preprocessing pieces fixed in the plan
def clean(a):
    a = a.copy()
    bad = (a[:, CI["m_lead"]] > 0.5) & (a[:, CI["g_lead"]] < 1.0) & (a[:, CI["v_lon"]] > 5) & (a[:, CI["m_valid"]] > 0)
    a[bad, CI["m_lead"]] = 0.0; a[bad, CI["g_lead"]] = 60.0; a[bad, CI["th_lead"]] = 10.0; a[bad, CI["dv_lead"]] = 0.0
    return a


def feat_osc(arrs):
    d = np.concatenate([a[a[:, CI["m_valid"]] > 0, CI["d_lat"]] for a in arrs]) if arrs else np.zeros(0)
    if len(d) < 32:
        return 0.0
    d = d - d.mean(); P = np.abs(np.fft.rfft(d * np.hanning(len(d)))) ** 2; f = np.fft.rfftfreq(len(d), BIN)
    return float(P[(f >= 0.25) & (f <= 0.6)].sum() / max(P[f > 0.02].sum(), 1e-12))


def feat_tg(arrs):
    a = np.concatenate(arrs) if arrs else np.zeros((0, len(CHANNELS) + 2))
    v = a[:, CI["m_valid"]] > 0
    if v.sum() == 0:
        return 0.0
    tg = v & (a[:, CI["m_lead"]] > 0.5) & (a[:, CI["th_lead"]] < 0.8) & (a[:, CI["v_lon"]] > 5)
    return float(tg.sum() / v.sum())


FEATFN = {"osc": feat_osc, "tg": feat_tg}
norm = L.AmpNorm.from_json(json.loads((AMPX / "model/norm.json").read_text()))


def load_arrays(ch, amp, keys):
    arr = L.to_arrays(L.attach_amplitude(ch[ch.track_key.isin(keys)], amp[amp.track_key.isin(keys)]))
    return {k: (clean(a) if CLEAN else a) for k, a in arr.items()}


def tensors(arr):
    return {k: torch.from_numpy(norm.apply(a)[0].T.copy()) for k, a in arr.items()}


def pool(z, seg, n):
    cnt = torch.zeros(n).index_add_(0, seg, torch.ones(len(seg)))[:, None]
    mean = torch.zeros(n, z.shape[1]).index_add_(0, seg, z) / cnt
    var = torch.zeros(n, z.shape[1]).index_add_(0, seg, (z - mean[seg]) ** 2) / cnt
    return torch.cat([mean, torch.sqrt(var + 1e-8)], 1)


K = len(FEATS)


class Det(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.enc, self.to_z = ae.enc, ae.to_z
        self.register_buffer("fmu", torch.zeros(16)); self.register_buffer("fsd", torch.ones(16))
        self.register_buffer("emu", torch.zeros(max(K, 1))); self.register_buffer("esd", torch.ones(max(K, 1)))
        self.head = nn.Sequential(nn.Linear(16 + K, 16), nn.GELU(), nn.Linear(16, 1))

    def codes(self, x):
        return self.to_z(self.enc(x).flatten(1))

    def head_in(self, f, e):
        h = (f - self.fmu) / self.fsd
        return torch.cat([h, (e - self.emu[:K]) / self.esd[:K]], 1) if K else h

    def forward(self, x, seg, n, e):
        return self.head(self.head_in(pool(self.codes(x), seg, n), e)).squeeze(1)


# ================================================================== phase 1: training data exactly as p12
lab = pd.read_parquet(DL / "data/processed/labels.parquet"); lab["track_key"] = lab.track_key.astype(str)
usable = lab[(lab.n_valid_bins >= 32) & lab.quality_ok.astype(bool)]
V = []
for (sid, vid), g in usable[(usable.group == "fp_normal") & (usable.role == "normal")].groupby(["scenario_id", "vehicle_id_oracle"]):
    sp = {"train": "train", "val_A": "val", "val_B": "val_B", "test": "test"}.get(g.split.iloc[0], "none")
    V.append({"vkey": f"{sid}|{vid}", "tracks": sorted(g.track_key), "y": 0, "set": sp, "source": "normal corpus"})
BV = pd.read_csv(CX / "data/batch_vehicles.csv")
for r in BV.itertuples():
    y = {"anomaly_verified": 1, "normal_unaffected_background": 0}.get(r.label)
    if y is None:
        continue
    V.append({"vkey": f"crossregion/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "y": y,
              "set": "train" if r.split_role == "train" else "val", "source": "cross-region batch", "anomaly_type": r.anomaly_type})
# ---- tailgating batches (plan: C_tailgating_data_20261008/PLAN_PREREGISTERED.md) ----
TGD = DL / "experiments_followup/tailgating_dataset_20261007"; NEWDATA = os.environ.get("NEWDATA") == "1"
LAB = {"anomaly_verified": 1, "normal_unaffected_background": 0, "normal_matched_control": 0}


def add_batch(csv, prefix, roles):
    for r in pd.read_csv(csv).itertuples():
        y = LAB.get(r.label)
        if r.split_role not in roles or y is None:
            continue
        V.append({"vkey": f"{prefix}/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "y": y,
                  "set": "train" if r.split_role == "train" else "val", "source": prefix,
                  "anomaly_type": r.anomaly_type if y == 1 else ("normal_matched_control" if r.label == "normal_matched_control" else "normal")})


add_batch(TGD / "data/batch_vehicles_v3.csv", "tailgating_v3", {"val_A"} | ({"train"} if NEWDATA else set()))   # tg3 val in BOTH arms
if NEWDATA:
    add_batch(TGD / "data/batch_vehicles_v4.csv", "tailgating_v4", {"train"})                            # v4 is train-only by plan
# ---- stall diversity / quantity arms (stall_diversity_20261008/plan/PLAN_PREREGISTERED.md) ----
SDD = DL / "experiments_followup/stall_diversity_20261008"; ARM = os.environ.get("STALLARM", "none"); assert ARM in ("none", "quantity", "diversity")
SB = pd.read_csv(SDD / "data/batch_vehicles.csv", dtype={"region": str})
if ARM != "none":
    for r in SB[SB.split_role == "train"].itertuples():
        y = {"anomaly_verified": 1, "normal_matched_control": 0, "normal_legit_stop": 0, "normal_unaffected_background": 0}.get(r.label)
        if y is None or r.arms not in ("both", ARM):
            continue
        if y == 0 and r.arms != "both":                  # addendum 2: normals only from shared runs -> identical normal set in both arms
            continue
        V.append({"vkey": f"stall_div/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "y": y, "set": "train", "source": "stall_div",
                  "anomaly_type": r.anomaly_type if y == 1 else (r.label if r.label != "normal_unaffected_background" else "normal"), "stall_mode": r.mode})
# ---- W batch (this round) ----
WARM = os.environ.get("ARM"); assert WARM in ("W1", "W2"), "ARM must be W1 or W2"; assert ARM == "diversity" and NEWDATA and CLEAN and FEATS == ["tg"]
WV = pd.read_csv(WDIR / "data/W_vehicles.csv", dtype={"region": str}); WV = WV[WV.tracks_usable.fillna("") != ""]
WLAB = {"anomaly_verified": 1, "normal_unaffected_background": 0, "normal_targeted_control": 0, "normal_matched_control": 0}
for r in WV.itertuples():
    y = WLAB.get(r.label)
    if y is None:
        continue
    src_ = "W_new" if y == 1 else ("W_control" if r.label == "normal_targeted_control" else ("W_tg_control" if r.label == "normal_matched_control" else "W_bg"))
    V.append({"vkey": f"W/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks_usable.split(";"), "y": y, "set": "train" if r.split == "train" else "val", "source": src_,
              "anomaly_type": f"anomaly_{r.type}" if y == 1 else (r.type if src_ == "W_control" else ("normal_matched_control" if y == 0 and src_ == "W_tg_control" else "normal"))})
for r in SB[(SB.split_role == "stallval") & (SB.label == "anomaly_verified")].itertuples():      # stallval stalls in the checkpoint validation (plan section 6)
    V.append({"vkey": f"stall_div/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "y": 1, "set": "val", "source": "stallval", "anomaly_type": r.anomaly_type})
V = pd.DataFrame(V); V = V[V.set != "none"]
keys = set(k for ks in V.tracks for k in ks)
ch = pd.read_parquet(DL / "data/processed/channels.parquet", columns=["track_key", "bin"] + CHANNELS + ["burn_in"]); ch["track_key"] = ch.track_key.astype(str)
BP = CX / "data/batch_processed"
ch = pd.concat([ch] + [pd.read_parquet(f, columns=["track_key", "bin"] + CHANNELS + ["burn_in"]) for f in sorted(BP.glob("*_channels.parquet"))], ignore_index=True)
ch["track_key"] = ch.track_key.astype(str)
amp = pd.concat([pd.read_parquet(AMPX / "data/amp_on_dl_bins_v2_consolidated.parquet")] + [pd.read_parquet(f) for f in sorted(BP.glob("*_amp.parquet"))])
amp["track_key"] = amp.track_key.astype(str)
for P_ in [TGD / "data/batch_processed_v3", TGD / "data/batch_processed_v4", SDD / "data/batch_processed", WDIR / "data/W_processed"]:
    if P_.exists():
        ch = pd.concat([ch] + [pd.read_parquet(f, columns=["track_key", "bin"] + CHANNELS + ["burn_in"]) for f in sorted(P_.glob("*_channels.parquet"))], ignore_index=True)
        amp = pd.concat([amp] + [pd.read_parquet(f) for f in sorted(P_.glob("*_amp.parquet"))], ignore_index=True)
ch["track_key"] = ch.track_key.astype(str); amp["track_key"] = amp.track_key.astype(str)
ARR = load_arrays(ch, amp, keys); X = tensors(ARR)


def vehicle_extras(vdf, arr):
    E = {}
    for r in vdf.itertuples():
        a = [arr[k] for k in r.tracks if k in arr]
        E[r.vkey] = [FEATFN[f](a) for f in FEATS]
    return E


EX = vehicle_extras(V, ARR); del ARR
win = {r.vkey: [(k, s) for k in r.tracks if k in X for s in window_starts(X[k].shape[1], T, STRIDE)] for r in V.itertuples()}
V["n_windows"] = V.vkey.map(lambda k: len(win[k])); V = V[V.n_windows >= 3]
say("vehicles (>=3 windows):", V.groupby(["set", "y"]).size().to_dict())


def window_tensor(Xd, k, s):
    x = Xd[k][:, s:s + T]
    return torch.cat([x, torch.zeros(x.shape[0], T - x.shape[1])], 1) if x.shape[1] < T else x


def batch(vkeys, maxw=None, rng=None):
    xs, seg = [], []
    for i, vk in enumerate(vkeys):
        ws = win[vk]
        if maxw and len(ws) > maxw:
            ws = [ws[j] for j in rng.choice(len(ws), maxw, replace=False)]
        xs += [window_tensor(X, k, s) for k, s in ws]; seg += [i] * len(ws)
    return torch.stack(xs), torch.tensor(seg)


def extras(vkeys, E):
    return torch.tensor([E[k] for k in vkeys], dtype=torch.float32) if K else torch.zeros(len(vkeys), 0)


ae = L.build_model(); st = torch.load(AMPX / "model/checkpoint_best.pt", map_location="cpu", weights_only=False); ae.load_state_dict(st["model"])
det = Det(ae)


@torch.no_grad()
def logits_of(vkeys, chunk=64):
    det.eval(); out, feats = [], []
    for i in range(0, len(vkeys), chunk):
        ks = vkeys[i:i + chunk]; x, seg = batch(ks)
        z = torch.cat([det.codes(x[j:j + 4096]) for j in range(0, len(x), 4096)]); f = pool(z, seg, len(ks)); feats.append(f)
        out.append(det.head(det.head_in(f, extras(ks, EX))).squeeze(1))
    return torch.cat(out).numpy(), torch.cat(feats).numpy()


tr = V[V.set == "train"]; trn, trp = tr[tr.y == 0].vkey.tolist(), tr[tr.y == 1].vkey.tolist(); va = V[V.set == "val"]
trp_orig = tr[(tr.y == 1) & (tr.source != "W_new")].vkey.tolist(); trp_new = tr[(tr.y == 1) & (tr.source == "W_new")].vkey.tolist()
trn_main = tr[(tr.y == 0) & (tr.source != "W_control")].vkey.tolist(); trn_ctl = tr[(tr.y == 0) & (tr.source == "W_control")].vkey.tolist()
say(json.dumps({"ARM": WARM, "positives_original": len(trp_orig), "positives_new": len(trp_new), "normals_main": len(trn_main), "normals_targeted_controls": len(trn_ctl)}))
_, F0 = logits_of(trn)
det.fmu.copy_(torch.tensor(F0.mean(0))); det.fsd.copy_(torch.tensor(F0.std(0) + 1e-6))
if K:
    E0 = np.array([EX[k] for k in trn]); det.emu.copy_(torch.tensor(E0.mean(0), dtype=torch.float32)); det.esd.copy_(torch.tensor(E0.std(0) + 1e-6, dtype=torch.float32))
    say("extra-feature stats on train normals:", dict(zip(FEATS, zip(E0.mean(0).round(4).tolist(), E0.std(0).round(4).tolist()))))
if not EVAL_ONLY:
    opt = torch.optim.AdamW([{"params": list(det.enc.parameters()) + list(det.to_z.parameters()), "lr": 1e-4},
                             {"params": det.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(); rng = np.random.default_rng(SEED)
    say("W train vehicles:", V[(V.set == "train") & V.source.str.startswith("W")].groupby(["source", "anomaly_type"]).size().to_dict()); say("added stall_div train vehicles:", V[V.source == "stall_div"].groupby(["y", "anomaly_type"]).size().to_dict() if "stall_div" in set(V.source) else {})
    typ = dict(zip(V.vkey, V.anomaly_type.fillna("normal"))); expo = []
    say(json.dumps({"train_normals": len(trn), "train_anomalies": len(trp), "val_normals": int((va.y == 0).sum()), "val_anomalies": int((va.y == 1).sum())}))
    best, best_ep, bad, hist = -1, -1, 0, []
    for ep in range(EPOCHS):
        t0 = time.time(); det.train(); tl = 0
        for _ in range(STEPS):
            if WARM == "W1":          # unchanged E_DIV sampler over the enlarged pools
                ks = list(rng.choice(trn, NB, replace=True)) + list(rng.choice(trp, PB, replace=True)); y = torch.tensor([0.] * NB + [1.] * PB)
            else:                     # W2: 36 + 4 normals, 16 original + 8 new positives (batch 64, positive share 37.5 %)
                ks = (list(rng.choice(trn_main, 36, replace=True)) + list(rng.choice(trn_ctl, 4, replace=True)) + list(rng.choice(trp_orig, 16, replace=True))
                      + list(rng.choice(trp_new, 8, replace=True))); y = torch.tensor([0.] * 40 + [1.] * 24)
            for k_ in ks:                                        # exposure log only (consumes no random numbers)
                expo.append((ep, typ[k_] if typ[k_] != "normal" else ("normal_W_bg" if k_.startswith("W/") else "normal_original")))
            x, seg = batch(ks, MAXW, rng)
            loss = lossf(det(x, seg, len(ks), extras(ks, EX)), y); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(det.parameters(), 5.0); opt.step(); tl += float(loss)
        lg, _ = logits_of(va.vkey.tolist()); ap = float(average_precision_score(va.y, lg)); auc = float(roc_auc_score(va.y, lg))
        imp = ap > best + 1e-6
        if imp:
            best, best_ep, bad = ap, ep, 0; torch.save({"model": det.state_dict(), "epoch": ep, "val_ap": ap, "feats": FEATS, "clean": CLEAN}, OUT / "model/checkpoint_best.pt")
        else:
            bad += 1
        hist.append({"epoch": ep, "train_loss": tl / STEPS, "val_ap": ap, "val_auc": auc, "improved": int(imp), "seconds": time.time() - t0})
        pd.DataFrame(hist).to_csv(OUT / "model/history.csv", index=False)
        pd.DataFrame(expo, columns=["epoch", "anomaly_type"]).value_counts().rename("draws").reset_index().sort_values(["epoch", "anomaly_type"]).to_csv(OUT / "model/exposure_per_epoch.csv", index=False)
        say(f"epoch {ep:2d} loss {tl / STEPS:.4f} val AP {ap:.4f} AUC {auc:.4f} {'*' if imp else ''} ({time.time() - t0:.0f}s)")
        if bad >= PATIENCE:
            break
    json.dump({"status": "done", "best_epoch": best_ep, "best_val_ap": best, "epochs_run": len(hist)}, open(OUT / "model/done.json", "w"), indent=1)

# ================================================================== phase 2: freeze thresholds on val_B (before evaluation data are loaded)
ck = torch.load(CKPT, map_location="cpu", weights_only=False); mk = det.load_state_dict(ck["model"], strict=False)
assert not mk.unexpected_keys and set(mk.missing_keys) <= {"emu", "esd"}, mk
det.eval(); best_ep, best = ck.get("epoch"), ck.get("val_ap")
vb = V[V.set == "val_B"].vkey.tolist(); lgb, _ = logits_of(vb)
if EVAL_ONLY:
    THR = json.loads(THRF.read_text())["thresholds_logit"]
else:
    THR = {"P99": float(np.percentile(lgb, 99)), "P95": float(np.percentile(lgb, 95))}
    json.dump({"thresholds_logit": THR, "n_valB_sufficient": len(vb), "checkpoint_epoch": best_ep, "val_ap": best,
               "rule": "numpy linear percentile of sufficient-evidence normal val_B logits; <3 windows = no decision"}, open(OUT / "model/FROZEN_THRESHOLDS.json", "w"), indent=1)
say("thresholds used:", THR)

vbk = V[V.set == "val_B"].vkey.tolist(); pd.DataFrame({"vkey": vbk, "logit": lgb}).to_csv(OUT / "model/valB_logits.csv", index=False)
say("phase 2 done: thresholds frozen on val_B; evaluation is done by scripts/w07_evaluate.py (frozen, common to E_DIV / W1 / W2)")
