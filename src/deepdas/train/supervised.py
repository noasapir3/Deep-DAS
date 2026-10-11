"""
Supervised training, threshold calibration and validation scoring of one seed of DeepDAS Ensemble v1.

Recipe (fixed):
  * input: 14 channels (13 track channels + DAS log-amplitude), lead-artefact cleaning, 64-bin windows with stride 8;
  * network: encoder of the amplitude ConvAE (initialised from models/deepdas_ensemble_v1/autoencoder_init.pt) -> 8-d window codes ->
    mean/std pooling per vehicle (16) standardised on the training normals + the standardised tailgating fraction -> MLP head (1 logit);
  * training data: the normal corpus (train split), the cross-region anomaly set, both tailgating sets and the stall/stop set;
    every batch = 48 normal + 16 anomalous vehicles drawn uniformly with replacement, at most 32 windows per vehicle;
  * AdamW (encoder 1e-4, head 1e-3, weight decay 1e-4), BCE-with-logits, gradient clipping 5, 120 steps x up to 25 epochs,
    early stopping (patience 6) on validation average precision; the best epoch is kept;
  * threshold: P99 of the logits of the sufficient-evidence (>= 3 windows) normal vehicles of the val_B regions.
Validation scoring writes one table with the logits of every validation group (normal test / val / val_B, the validation anomalies,
the stall validation runs, the new-scene check runs and the nine development cases).

Determinism: with the same seed, data, library versions and NTHREADS=1 the run is reproduced exactly (verified for all five seeds).
Data identifiers (vehicle keys, track keys, run ids) are the original ones.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .. import paths
from ..amplitude import AmpNorm, attach_amplitude, build_model, to_arrays
from ..data.channels import CHANNELS
from ..data.dataset import window_starts
from ..ensemble.detector import Detector, clean_lead_artifacts as clean, pool, tailgating_fraction

T, STRIDE, MAXW, NB, PB, STEPS, EPOCHS, PATIENCE = 64, 8, 32, 48, 16, 120, 25, 6
FEATS = ["tg"]
K = len(FEATS)
CH_COLS = ["track_key", "bin"] + list(CHANNELS) + ["burn_in"]
SETS = paths.ANOMALY_SETS
LAB_BATCH = {"anomaly_verified": 1, "normal_unaffected_background": 0, "normal_matched_control": 0}
LAB_STALL = {"anomaly_verified": 1, "normal_matched_control": 0, "normal_legit_stop": 0, "normal_unaffected_background": 0}
DEV_CASES = {"speeding": "region_003/anomaly_speeding|veh001", "C1 real braking": "C1_hard_braking|veh001", "stalled": "region_020/anomaly_stalled_vehicle|veh001",
             "C4 collision striker": "C4_collision_contact_REUSED|veh001", "reckless": "region_120/anomaly_reckless_driver|veh001",
             "weaving": "region_037/anomaly_lane_weaving|veh001", "overweight": "region_140/anomaly_overweight_vehicle|veh001",
             "C2 straddling": "C2_lane_straddling|veh001", "C3 tailgating": "C3_tailgating|veh001"}
CONTROLLED = {"C1_hard_braking": "controlled_hard_braking", "C2_lane_straddling": "controlled_lane_straddling", "C3_tailgating": "controlled_tailgating",
              "C4_collision_contact_REUSED": "controlled_collision"}


# ------------------------------------------------------------------ data
def vehicle_table() -> pd.DataFrame:
    """All vehicles of the training / validation / threshold sets, in the fixed order used by every run (the order matters for sampling)."""
    lab = pd.read_parquet(paths.PROCESSED / "labels.parquet"); lab["track_key"] = lab.track_key.astype(str)
    usable = lab[(lab.n_valid_bins >= 32) & lab.quality_ok.astype(bool)]; V = []
    for (sid, vid), g in usable[(usable.group == "fp_normal") & (usable.role == "normal")].groupby(["scenario_id", "vehicle_id_oracle"]):
        sp = {"train": "train", "val_A": "val", "val_B": "val_B", "test": "test"}.get(g.split.iloc[0], "none")
        V.append({"vkey": f"{sid}|{vid}", "tracks": sorted(g.track_key), "y": 0, "set": sp, "source": "normal corpus"})
    for r in pd.read_csv(SETS / "cross_region/vehicles.csv").itertuples():
        y = {"anomaly_verified": 1, "normal_unaffected_background": 0}.get(r.label)
        if y is None:
            continue
        V.append({"vkey": f"crossregion/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "y": y,
                  "set": "train" if r.split_role == "train" else "val", "source": "cross-region batch", "anomaly_type": r.anomaly_type})

    def add_batch(csv, prefix, roles):
        for r in pd.read_csv(csv).itertuples():
            y = LAB_BATCH.get(r.label)
            if r.split_role not in roles or y is None:
                continue
            V.append({"vkey": f"{prefix}/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "y": y,
                      "set": "train" if r.split_role == "train" else "val", "source": prefix,
                      "anomaly_type": r.anomaly_type if y == 1 else ("normal_matched_control" if r.label == "normal_matched_control" else "normal")})

    add_batch(SETS / "tailgating_routes/vehicles.csv", "tailgating_v3", {"val_A", "train"})
    add_batch(SETS / "tailgating_matched_controls/vehicles.csv", "tailgating_v4", {"train"})
    SB = pd.read_csv(SETS / "stalls_and_stops/vehicles.csv", dtype={"region": str})
    for r in SB[SB.split_role == "train"].itertuples():
        y = LAB_STALL.get(r.label)
        if y is None or r.arms not in ("both", "diversity"):
            continue
        if y == 0 and r.arms != "both":
            continue
        V.append({"vkey": f"stall_div/{r.run_id}|{r.vehicle_id_oracle}", "tracks": r.tracks.split(";"), "y": y, "set": "train", "source": "stall_div",
                  "anomaly_type": r.anomaly_type if y == 1 else (r.label if r.label != "normal_unaffected_background" else "normal"), "stall_mode": r.mode})
    V = pd.DataFrame(V)
    return V[V.set != "none"]


def load_channels_and_amplitude():
    """Channels and amplitude of the normal corpus and of the training sets (concatenated in the fixed order)."""
    ch = pd.read_parquet(paths.PROCESSED / "channels.parquet", columns=CH_COLS); ch["track_key"] = ch.track_key.astype(str)
    files = lambda d, pat: sorted((SETS / d / "runs").glob(pat))
    ch = pd.concat([ch] + [pd.read_parquet(f, columns=CH_COLS) for f in files("cross_region", "*_channels.parquet")], ignore_index=True)
    ch["track_key"] = ch.track_key.astype(str)
    amp = pd.concat([pd.read_parquet(paths.PROCESSED / "amplitude.parquet")] + [pd.read_parquet(f) for f in files("cross_region", "*_amp.parquet")])
    amp["track_key"] = amp.track_key.astype(str)
    for d in ("tailgating_routes", "tailgating_matched_controls", "stalls_and_stops"):
        ch = pd.concat([ch] + [pd.read_parquet(f, columns=CH_COLS) for f in files(d, "*_channels.parquet")], ignore_index=True)
        amp = pd.concat([amp] + [pd.read_parquet(f) for f in files(d, "*_amp.parquet")], ignore_index=True)
    ch["track_key"] = ch.track_key.astype(str); amp["track_key"] = amp.track_key.astype(str)
    return ch, amp


class SeedRun:
    """One seed: data tensors, windows, the detector and the train / calibrate / evaluate steps."""

    def __init__(self, seed: int, out_dir: Path, threads: int = 1):
        self.seed, self.out = seed, Path(out_dir); (self.out / "model").mkdir(parents=True, exist_ok=True); (self.out / "evaluation").mkdir(exist_ok=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True); torch.set_num_threads(threads)
        self.log = open(self.out / "model/train_log.txt", "a")
        self.norm = AmpNorm.from_json(json.loads((paths.ENSEMBLE / "input_normalization.json").read_text()))
        V = vehicle_table(); keys = set(k for ks in V.tracks for k in ks)
        self.ch, self.amp = load_channels_and_amplitude()
        arr = self.load_arrays(self.ch, self.amp, keys); self.X = self.tensors(arr); self.EX = self.vehicle_extras(V, arr); del arr
        self.win = {r.vkey: [(k, s) for k in r.tracks if k in self.X for s in window_starts(self.X[k].shape[1], T, STRIDE)] for r in V.itertuples()}
        V["n_windows"] = V.vkey.map(lambda k: len(self.win[k])); self.V = V[V.n_windows >= 3]
        self.say("vehicles (>=3 windows):", self.V.groupby(["set", "y"]).size().to_dict())
        ae = build_model(); st = torch.load(paths.ENSEMBLE / "autoencoder_init.pt", map_location="cpu", weights_only=False); ae.load_state_dict(st["model"])
        self.det = Detector(ae, K)
        tr = self.V[self.V.set == "train"]; self.trn, self.trp = tr[tr.y == 0].vkey.tolist(), tr[tr.y == 1].vkey.tolist(); self.va = self.V[self.V.set == "val"]
        _, F0 = self.logits_of(self.trn)
        self.det.fmu.copy_(torch.tensor(F0.mean(0))); self.det.fsd.copy_(torch.tensor(F0.std(0) + 1e-6))
        E0 = np.array([self.EX[k] for k in self.trn])
        self.det.emu.copy_(torch.tensor(E0.mean(0), dtype=torch.float32)); self.det.esd.copy_(torch.tensor(E0.std(0) + 1e-6, dtype=torch.float32))

    # ---- helpers
    def say(self, *a):
        s = " ".join(str(x) for x in a); print(s, flush=True); self.log.write(s + "\n"); self.log.flush()

    def load_arrays(self, ch, amp, keys):
        arr = to_arrays(attach_amplitude(ch[ch.track_key.isin(keys)], amp[amp.track_key.isin(keys)]))
        return {k: clean(a) for k, a in arr.items()}

    def tensors(self, arr):
        return {k: torch.from_numpy(self.norm.apply(a)[0].T.copy()) for k, a in arr.items()}

    def vehicle_extras(self, vdf, arr):
        return {r.vkey: [tailgating_fraction([arr[k] for k in r.tracks if k in arr])] for r in vdf.itertuples()}

    def window_tensor(self, k, s):
        x = self.X[k][:, s:s + T]
        return torch.cat([x, torch.zeros(x.shape[0], T - x.shape[1])], 1) if x.shape[1] < T else x

    def batch(self, vkeys, maxw=None, rng=None):
        xs, seg = [], []
        for i, vk in enumerate(vkeys):
            ws = self.win[vk]
            if maxw and len(ws) > maxw:
                ws = [ws[j] for j in rng.choice(len(ws), maxw, replace=False)]
            xs += [self.window_tensor(k, s) for k, s in ws]; seg += [i] * len(ws)
        return torch.stack(xs), torch.tensor(seg)

    def extras(self, vkeys):
        return torch.tensor([self.EX[k] for k in vkeys], dtype=torch.float32)

    @torch.no_grad()
    def logits_of(self, vkeys, chunk=64):
        self.det.eval(); out, feats = [], []
        for i in range(0, len(vkeys), chunk):
            ks = vkeys[i:i + chunk]; x, seg = self.batch(ks)
            z = torch.cat([self.det.codes(x[j:j + 4096]) for j in range(0, len(x), 4096)]); f = pool(z, seg, len(ks)); feats.append(f)
            out.append(self.det.head(self.det.head_in(f, self.extras(ks))).squeeze(1))
        return torch.cat(out).numpy(), torch.cat(feats).numpy()

    # ---- steps
    def train(self):
        det = self.det
        opt = torch.optim.AdamW([{"params": list(det.enc.parameters()) + list(det.to_z.parameters()), "lr": 1e-4},
                                 {"params": det.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(); rng = np.random.default_rng(self.seed)
        typ = dict(zip(self.V.vkey, self.V.anomaly_type.fillna("normal"))); expo = []; va = self.va
        self.say(json.dumps({"train_normals": len(self.trn), "train_anomalies": len(self.trp), "val_normals": int((va.y == 0).sum()), "val_anomalies": int((va.y == 1).sum())}))
        best, best_ep, bad, hist = -1, -1, 0, []
        for ep in range(EPOCHS):
            t0 = time.time(); det.train(); tl = 0
            for _ in range(STEPS):
                ks = list(rng.choice(self.trn, NB, replace=True)) + list(rng.choice(self.trp, PB, replace=True)); y = torch.tensor([0.] * NB + [1.] * PB)
                for k_ in ks[NB:]:
                    expo.append((ep, typ[k_]))
                x, seg = self.batch(ks, MAXW, rng)
                loss = lossf(det(x, seg, len(ks), self.extras(ks)), y); opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(det.parameters(), 5.0); opt.step(); tl += float(loss)
            lg, _ = self.logits_of(va.vkey.tolist()); ap = float(average_precision_score(va.y, lg)); auc = float(roc_auc_score(va.y, lg))
            imp = ap > best + 1e-6
            if imp:
                best, best_ep, bad = ap, ep, 0
                torch.save({"model": det.state_dict(), "epoch": ep, "val_ap": ap, "feats": FEATS, "clean": True}, self.out / "model/checkpoint_best.pt")
            else:
                bad += 1
            hist.append({"epoch": ep, "train_loss": tl / STEPS, "val_ap": ap, "val_auc": auc, "improved": int(imp), "seconds": time.time() - t0})
            pd.DataFrame(hist).to_csv(self.out / "model/history.csv", index=False)
            pd.DataFrame(expo, columns=["epoch", "anomaly_type"]).value_counts().rename("draws").reset_index().sort_values(["epoch", "anomaly_type"]).to_csv(
                self.out / "model/exposure_per_epoch.csv", index=False)
            self.say(f"epoch {ep:2d} loss {tl / STEPS:.4f} val AP {ap:.4f} AUC {auc:.4f} {'*' if imp else ''} ({time.time() - t0:.0f}s)")
            if bad >= PATIENCE:
                break
        json.dump({"status": "done", "best_epoch": best_ep, "best_val_ap": best, "epochs_run": len(hist)}, open(self.out / "model/done.json", "w"), indent=1)

    def load_checkpoint(self, ckpt: Path):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False); mk = self.det.load_state_dict(ck["model"], strict=False)
        assert not mk.unexpected_keys and set(mk.missing_keys) <= {"emu", "esd"}, mk
        self.det.eval(); return ck

    def calibrate(self, ckpt: Path, frozen_thresholds: Path | None = None) -> dict:
        """Threshold on the val_B normals (or the frozen thresholds when given). Writes FROZEN_THRESHOLDS.json only when calibrating."""
        ck = self.load_checkpoint(ckpt)
        self.vb = self.V[self.V.set == "val_B"].vkey.tolist(); self.lgb, _ = self.logits_of(self.vb)
        if frozen_thresholds is not None:
            self.THR = json.loads(Path(frozen_thresholds).read_text())["thresholds_logit"]
        else:
            self.THR = {"P99": float(np.percentile(self.lgb, 99)), "P95": float(np.percentile(self.lgb, 95))}
            json.dump({"thresholds_logit": self.THR, "n_valB_sufficient": len(self.vb), "checkpoint_epoch": ck.get("epoch"), "val_ap": ck.get("val_ap"),
                       "rule": "numpy linear percentile of sufficient-evidence normal val_B logits; <3 windows = no decision"},
                      open(self.out / "model/FROZEN_THRESHOLDS.json", "w"), indent=1)
        return self.THR

    def _score_external(self, ch_, amp_, vehicles):
        ks = set(k for t in vehicles.tracks for k in t); arr = self.load_arrays(ch_, amp_, ks)
        self.X = self.tensors(arr); self.EX = self.vehicle_extras(vehicles, arr)
        self.win = {r.vkey: [(k, s) for k in r.tracks if k in self.X for s in window_starts(self.X[k].shape[1], T, STRIDE)] for r in vehicles.itertuples()}
        ok = [k for k in vehicles.vkey if len(self.win[k]) > 0]; lg, _ = self.logits_of(ok); return {k: (float(l), len(self.win[k])) for k, l in zip(ok, lg)}

    def evaluate(self) -> pd.DataFrame:
        """Validation scoring (after calibrate). Writes evaluation/vehicle_scores.csv."""
        V, THR = self.V, self.THR; dec = lambda lg, n, t: "no decision" if n < 3 else ("ALARM" if lg > t else "no"); rows = []
        for sp in ("test", "val"):
            ks = V[(V.set == sp) & (V.y == 0) & (V.source == "normal corpus")].vkey.tolist(); lg, _ = self.logits_of(ks)
            rows += [{"group": f"normal_{sp}", "vkey": k, "logit": float(l)} for k, l in zip(ks, lg)]
        rows += [{"group": "normal_val_B", "vkey": k, "logit": float(l)} for k, l in zip(self.vb, self.lgb)]
        va = self.va; lgv, _ = self.logits_of(va.vkey.tolist())
        rows += [{"group": "validation", "vkey": k, "logit": float(l), "y": int(y), "anomaly_type": t, "source": so}
                 for k, l, y, t, so in zip(va.vkey, lgv, va.y, va.anomaly_type.fillna("normal"), va.source)]
        lab = pd.read_parquet(paths.PROCESSED / "labels.parquet"); lab["track_key"] = lab.track_key.astype(str)
        usable = lab[(lab.n_valid_bins >= 32) & lab.quality_ok.astype(bool)]
        orig = usable[usable.group == "fp_anomaly"].groupby(["scenario_id", "vehicle_id_oracle"]).track_key.apply(sorted).reset_index().rename(columns={"track_key": "tracks"})
        orig["vkey"] = orig.scenario_id + "|" + orig.vehicle_id_oracle
        res = self._score_external(self.ch, self.amp, orig); cc, ca, cv = [], [], []
        for run, stem in CONTROLLED.items():
            D = paths.DEVELOPMENT_CASES
            c = pd.read_parquet(D / f"{stem}_channels.parquet"); m = pd.read_parquet(D / f"{stem}_track_meta.parquet"); a = pd.read_parquet(D / f"{stem}_amplitude.parquet")
            for d_ in (c, m, a):
                d_["track_key"] = run + "::" + d_.track_key.astype(str)
            cc.append(c); ca.append(a)
            vv = m[m.usable].groupby("vehicle_id_oracle").track_key.apply(sorted).reset_index().rename(columns={"track_key": "tracks"}); vv["vkey"] = run + "|" + vv.vehicle_id_oracle; cv.append(vv)
        res.update(self._score_external(pd.concat(cc, ignore_index=True), pd.concat(ca, ignore_index=True), pd.concat(cv, ignore_index=True)))
        nb = pd.read_csv(SETS / "new_scene_check/vehicles.csv"); nb = nb[nb.label.isin(["anomaly_verified", "normal_unaffected_background"])].copy()
        nb["vkey"] = "newscene/" + nb.run_id + "|" + nb.vehicle_id_oracle; nb["tracks"] = nb.tracks.str.split(";")
        nr = sorted((SETS / "new_scene_check/runs").glob("*_channels.parquet")); na_ = sorted((SETS / "new_scene_check/runs").glob("*_amp.parquet"))
        nc = pd.concat([pd.read_parquet(f, columns=CH_COLS) for f in nr]); na = pd.concat([pd.read_parquet(f) for f in na_])
        nc["track_key"] = nc.track_key.astype(str); na["track_key"] = na.track_key.astype(str)
        res.update(self._score_external(nc, na, nb))
        for r in nb.itertuples():
            if r.vkey in res:
                rows.append({"group": "newscene_anomaly" if r.label == "anomaly_verified" else "newscene_background", "vkey": r.vkey, "logit": res[r.vkey][0],
                             "n_windows": res[r.vkey][1], "anomaly_type": r.anomaly_type})
        SB = pd.read_csv(SETS / "stalls_and_stops/vehicles.csv", dtype={"region": str})
        SVV = SB[(SB.split_role == "stallval") & SB.label.isin(["anomaly_verified", "normal_matched_control", "normal_legit_stop", "normal_unaffected_background"])].copy()
        SVV["vkey"] = "stallval/" + SVV.run_id + "|" + SVV.vehicle_id_oracle; SVV["tracks"] = SVV.tracks.str.split(";")
        sd = SETS / "stalls_and_stops/runs"
        SC_ = pd.concat([pd.read_parquet(f, columns=CH_COLS) for f in sorted(sd.glob("st_st_*_channels.parquet"))])
        SA_ = pd.concat([pd.read_parquet(f) for f in sorted(sd.glob("st_st_*_amp.parquet"))])
        SC_["track_key"] = SC_.track_key.astype(str); SA_["track_key"] = SA_.track_key.astype(str)
        res.update(self._score_external(SC_, SA_, SVV))
        for r in SVV.itertuples():
            rows.append({"group": "stallval", "vkey": r.vkey, "logit": res[r.vkey][0] if r.vkey in res else np.nan, "n_windows": res[r.vkey][1] if r.vkey in res else 0,
                         "y": int(r.label == "anomaly_verified"), "anomaly_type": r.label, "stall_mode": r.mode, "sensing": r.sensing})
        for nm, k in DEV_CASES.items():
            rows.append({"group": "valid9", "case": nm, "vkey": k, "logit": res[k][0], "n_windows": res[k][1]})
        S = pd.DataFrame(rows); S["n_windows"] = S.n_windows.fillna(3)
        for op in ("P99", "P95"):
            S[op] = [dec(l, n, THR[op]) for l, n in zip(S.logit, S.n_windows)]
        S.to_csv(self.out / "evaluation/vehicle_scores.csv", index=False)
        return S


def _legitimate_stop_set(run: "SeedRun") -> pd.DataFrame:
    """Scores of the legitimate-stop validation set (stop study in val_A regions: legitimate stops 1-10 s, matched controls, abnormal stops)."""
    SB2 = pd.read_csv(SETS / "legitimate_stop_validation/vehicles.csv", dtype={"region": str})
    SV2 = SB2[SB2.label.isin(["anomaly_verified", "normal_matched_control", "normal_legit_stop", "normal_unaffected_background"])].copy()
    SV2["vkey"] = "stopval2/" + SV2.run_id + "|" + SV2.vehicle_id_oracle; SV2["tracks"] = SV2.tracks.str.split(";")
    d = SETS / "legitimate_stop_validation/runs"
    SC2 = pd.concat([pd.read_parquet(f, columns=CH_COLS) for f in sorted(d.glob("sv2_*_channels.parquet"))])
    SA2 = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("sv2_*_amp.parquet"))])
    SC2["track_key"] = SC2.track_key.astype(str); SA2["track_key"] = SA2.track_key.astype(str)
    res = run._score_external(SC2, SA2, SV2); rows = []
    for r in SV2.itertuples():
        rows.append({"group": "stopval2", "vkey": r.vkey, "logit": res[r.vkey][0] if r.vkey in res else np.nan, "n_windows": res[r.vkey][1] if r.vkey in res else 0,
                     "y": int(r.label == "anomaly_verified"), "anomaly_type": r.label, "stall_mode": r.mode, "sensing": r.sensing, "obs": r.obs, "hold_s": r.hold_s})
    S = pd.DataFrame(rows); S.to_csv(run.out / "evaluation/vehicle_scores_legitimate_stop_set.csv", index=False)
    return S


def train_seed(seed: int, out_dir: Path, threads: int = 1) -> Path:
    """Train one seed, calibrate its threshold on val_B and score the validation groups."""
    run = SeedRun(seed, out_dir, threads)
    if (Path(out_dir) / "model/done.json").exists():
        raise SystemExit(f"{out_dir} already holds a trained model; refusing to overwrite")
    run.train(); run.calibrate(Path(out_dir) / "model/checkpoint_best.pt"); run.evaluate()
    return Path(out_dir) / "model/checkpoint_best.pt"


def evaluate_frozen_seed(seed: int, out_dir: Path, threads: int = 1) -> pd.DataFrame:
    """Validation scoring of a frozen seed of DeepDAS Ensemble v1 (frozen checkpoint and thresholds; nothing is trained or calibrated)."""
    run = SeedRun(seed, out_dir, threads); d = paths.ENSEMBLE / f"seed_{seed}"
    run.calibrate(d / "checkpoint.pt", d / "thresholds.json")
    S = run.evaluate(); _legitimate_stop_set(run)
    return S
