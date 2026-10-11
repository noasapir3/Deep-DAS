#!/usr/bin/env python3
"""
Stall diversity / quantity arms (../stall_diversity_20261008/plan/PLAN_PREREGISTERED.md): x_train_eval_c.py + STALLARM env (none|quantity|diversity),
an exposure log and the stallval evaluation group; everything else unchanged. Was: Experiments A–D (see PLAN_PREREGISTERED.md).  One script, one run per call:
    python x_train_eval.py <run_name> <clean 0|1> <feats comma list or none> <out_dir>
Phase 1 trains (p12 recipe, unchanged), phase 2 freezes val_B thresholds, phase 3 evaluates.  Writes only inside <out_dir>.
"""
import json, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score

RUN, CLEAN, FEATS, OUT = sys.argv[1], sys.argv[2] == "1", [f for f in sys.argv[3].split(",") if f and f != "none"], Path(sys.argv[4]).resolve()
HERE = Path(__file__).resolve().parent; DL = HERE.parents[1]
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
SDD = HERE.parent / "stall_diversity_20261008"; ARM = os.environ.get("STALLARM", "none"); assert ARM in ("none", "quantity", "diversity")
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
V = pd.DataFrame(V); V = V[V.set != "none"]
keys = set(k for ks in V.tracks for k in ks)
ch = pd.read_parquet(DL / "data/processed/channels.parquet", columns=["track_key", "bin"] + CHANNELS + ["burn_in"]); ch["track_key"] = ch.track_key.astype(str)
BP = CX / "data/batch_processed"
ch = pd.concat([ch] + [pd.read_parquet(f, columns=["track_key", "bin"] + CHANNELS + ["burn_in"]) for f in sorted(BP.glob("*_channels.parquet"))], ignore_index=True)
ch["track_key"] = ch.track_key.astype(str)
amp = pd.concat([pd.read_parquet(AMPX / "data/amp_on_dl_bins_v2_consolidated.parquet")] + [pd.read_parquet(f) for f in sorted(BP.glob("*_amp.parquet"))])
amp["track_key"] = amp.track_key.astype(str)
for P_ in [TGD / "data/batch_processed_v3", TGD / "data/batch_processed_v4", SDD / "data/batch_processed"]:
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
_, F0 = logits_of(trn)
det.fmu.copy_(torch.tensor(F0.mean(0))); det.fsd.copy_(torch.tensor(F0.std(0) + 1e-6))
if K:
    E0 = np.array([EX[k] for k in trn]); det.emu.copy_(torch.tensor(E0.mean(0), dtype=torch.float32)); det.esd.copy_(torch.tensor(E0.std(0) + 1e-6, dtype=torch.float32))
    say("extra-feature stats on train normals:", dict(zip(FEATS, zip(E0.mean(0).round(4).tolist(), E0.std(0).round(4).tolist()))))
if not EVAL_ONLY:
    opt = torch.optim.AdamW([{"params": list(det.enc.parameters()) + list(det.to_z.parameters()), "lr": 1e-4},
                             {"params": det.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(); rng = np.random.default_rng(SEED)
    say("added stall_div train vehicles:", V[V.source == "stall_div"].groupby(["y", "anomaly_type"]).size().to_dict() if "stall_div" in set(V.source) else {})
    typ = dict(zip(V.vkey, V.anomaly_type.fillna("normal"))); expo = []
    say(json.dumps({"train_normals": len(trn), "train_anomalies": len(trp), "val_normals": int((va.y == 0).sum()), "val_anomalies": int((va.y == 1).sum())}))
    best, best_ep, bad, hist = -1, -1, 0, []
    for ep in range(EPOCHS):
        t0 = time.time(); det.train(); tl = 0
        for _ in range(STEPS):
            ks = list(rng.choice(trn, NB, replace=True)) + list(rng.choice(trp, PB, replace=True)); y = torch.tensor([0.] * NB + [1.] * PB)
            for k_ in ks[NB:]:                                   # exposure log only (consumes no random numbers)
                expo.append((ep, typ[k_]))
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

# ================================================================== phase 3: evaluation
dec = lambda lg, n, t: "no decision" if n < 3 else ("ALARM" if lg > t else "no")
rows = []
for sp in ("test", "val"):
    ks = V[(V.set == sp) & (V.y == 0) & (V.source == "normal corpus")].vkey.tolist(); lg, _ = logits_of(ks)
    rows += [{"group": f"normal_{sp}", "vkey": k, "logit": float(l)} for k, l in zip(ks, lg)]
rows += [{"group": "normal_val_B", "vkey": k, "logit": float(l)} for k, l in zip(vb, lgb)]
lgv, _ = logits_of(va.vkey.tolist())
rows += [{"group": "validation", "vkey": k, "logit": float(l), "y": int(y), "anomaly_type": t, "source": so}
         for k, l, y, t, so in zip(va.vkey, lgv, va.y, va.anomaly_type.fillna("normal"), va.source)]


def score_external(ch_, amp_, vehicles):
    global X, win, EX
    ks = set(k for t in vehicles.tracks for k in t); arr = load_arrays(ch_, amp_, ks)
    X = tensors(arr); EX = vehicle_extras(vehicles, arr)
    win = {r.vkey: [(k, s) for k in r.tracks if k in X for s in window_starts(X[k].shape[1], T, STRIDE)] for r in vehicles.itertuples()}
    ok = [k for k in vehicles.vkey if len(win[k]) > 0]; lg, _ = logits_of(ok)
    return {k: (float(l), len(win[k])) for k, l in zip(ok, lg)}


orig = usable[usable.group == "fp_anomaly"].groupby(["scenario_id", "vehicle_id_oracle"]).track_key.apply(sorted).reset_index().rename(columns={"track_key": "tracks"})
orig["vkey"] = orig.scenario_id + "|" + orig.vehicle_id_oracle
res = score_external(ch, amp, orig)
R = AMPX / "data/r1_runs"; cc, ca, cv = [], [], []
for run in ["C1_hard_braking", "C2_lane_straddling", "C3_tailgating", "C4_collision_contact_REUSED"]:
    c = pd.read_parquet(R / f"{run}_channels.parquet"); m = pd.read_parquet(R / f"{run}_track_meta.parquet"); a = pd.read_parquet(R / f"{run}_amp.parquet")
    for d_ in (c, m, a):
        d_["track_key"] = run + "::" + d_.track_key.astype(str)
    cc.append(c); ca.append(a)
    vv = m[m.usable].groupby("vehicle_id_oracle").track_key.apply(sorted).reset_index().rename(columns={"track_key": "tracks"}); vv["vkey"] = run + "|" + vv.vehicle_id_oracle; cv.append(vv)
res.update(score_external(pd.concat(cc, ignore_index=True), pd.concat(ca, ignore_index=True), pd.concat(cv, ignore_index=True)))
NSD = DL / "experiments_followup/newscene_test_20261007"
if (NSD / "data/batch_vehicles.csv").exists():
    nb = pd.read_csv(NSD / "data/batch_vehicles.csv"); nb = nb[nb.label.isin(["anomaly_verified", "normal_unaffected_background"])].copy()
    nb["vkey"] = "newscene/" + nb.run_id + "|" + nb.vehicle_id_oracle; nb["tracks"] = nb.tracks.str.split(";")
    nc = pd.concat([pd.read_parquet(f, columns=["track_key", "bin"] + CHANNELS + ["burn_in"]) for f in sorted((NSD / "data/batch_processed").glob("*_channels.parquet"))])
    na = pd.concat([pd.read_parquet(f) for f in sorted((NSD / "data/batch_processed").glob("*_amp.parquet"))])
    nc["track_key"] = nc.track_key.astype(str); na["track_key"] = na.track_key.astype(str)
    res.update(score_external(nc, na, nb))
    for r in nb.itertuples():
        if r.vkey in res:
            rows.append({"group": "newscene_anomaly" if r.label == "anomaly_verified" else "newscene_background", "vkey": r.vkey, "logit": res[r.vkey][0],
                         "n_windows": res[r.vkey][1], "anomaly_type": r.anomaly_type})
SVV = SB[(SB.split_role == "stallval") & SB.label.isin(["anomaly_verified", "normal_matched_control", "normal_legit_stop", "normal_unaffected_background"])].copy()
SVV["vkey"] = "stallval/" + SVV.run_id + "|" + SVV.vehicle_id_oracle; SVV["tracks"] = SVV.tracks.str.split(";")
SC_ = pd.concat([pd.read_parquet(f, columns=["track_key", "bin"] + CHANNELS + ["burn_in"]) for f in sorted((SDD / "data/batch_processed").glob("st_st_*_channels.parquet"))])
SA_ = pd.concat([pd.read_parquet(f) for f in sorted((SDD / "data/batch_processed").glob("st_st_*_amp.parquet"))])
SC_["track_key"] = SC_.track_key.astype(str); SA_["track_key"] = SA_.track_key.astype(str)
res.update(score_external(SC_, SA_, SVV))
for r in SVV.itertuples():
    rows.append({"group": "stallval", "vkey": r.vkey, "logit": res[r.vkey][0] if r.vkey in res else np.nan, "n_windows": res[r.vkey][1] if r.vkey in res else 0,
                 "y": int(r.label == "anomaly_verified"), "anomaly_type": r.label, "stall_mode": r.mode, "sensing": r.sensing})
VALID9 = {"speeding": "region_003/anomaly_speeding|veh001", "C1 real braking": "C1_hard_braking|veh001", "stalled": "region_020/anomaly_stalled_vehicle|veh001",
          "C4 collision striker": "C4_collision_contact_REUSED|veh001", "reckless": "region_120/anomaly_reckless_driver|veh001",
          "weaving": "region_037/anomaly_lane_weaving|veh001", "overweight": "region_140/anomaly_overweight_vehicle|veh001",
          "C2 straddling": "C2_lane_straddling|veh001", "C3 tailgating": "C3_tailgating|veh001"}
for nm, k in VALID9.items():
    rows.append({"group": "valid9", "case": nm, "vkey": k, "logit": res[k][0], "n_windows": res[k][1]})
S = pd.DataFrame(rows)
S["n_windows"] = S.n_windows.fillna(3)
for op in ("P99", "P95"):
    S[op] = [dec(l, n, THR[op]) for l, n in zip(S.logit, S.n_windows)]
S.to_csv(OUT / "evaluation/vehicle_scores.csv", index=False)
say("evaluation written:", S.group.value_counts().to_dict())
