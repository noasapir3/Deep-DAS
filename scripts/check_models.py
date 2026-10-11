#!/usr/bin/env python3
"""Installation check that needs no data package (a few seconds, CPU).

It does not score any vehicle and produces no result. It checks that:
  1. the five checkpoints of DeepDAS Ensemble v1 match the sha256 recorded in models/deepdas_ensemble_v1/ensemble.json;
  2. they load with the installed PyTorch, with the expected architecture (14 input channels, head 17 -> 16 -> 1);
  3. the two inference options behave as documented for the 17th head input x17:
       active       x17 = (phi_u - mu_phi) / sigma_phi        (the model as trained, selected and tested)
       neutralized  x17 = 0                                    (post-hoc check on the same weights)
     while the 16 learned head inputs are identical in both options.
The probe values below are arbitrary test inputs for this check, not data.
Usage: python scripts/check_models.py
"""
import hashlib, json, os, sys
from pathlib import Path
import _setup  # noqa: F401
import torch
from deepdas import paths
from deepdas.ensemble.detector import load_detector

ok = True


def report(name, passed, detail=""):
    global ok
    ok &= bool(passed); print(f"[{'PASS' if passed else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


spec = json.loads((paths.ENSEMBLE / "ensemble.json").read_text())
for s in spec["seeds"]:
    p = paths.ENSEMBLE / s["checkpoint"]
    report(f"checkpoint seed {s['seed']} sha256", hashlib.sha256(p.read_bytes()).hexdigest() == s["checkpoint_sha256"], s["checkpoint_sha256"][:12])


def load(mode):
    os.environ["DEEPDAS_PHI_MODE"] = mode
    return [load_detector(paths.ENSEMBLE / s["checkpoint"], 1) for s in spec["seeds"]]


act, neu = load("active"), load("neutralized")
d = act[0]
report("architecture", d.head[0].in_features == 17 and d.head[2].out_features == 1 and sum(p.numel() for p in d.parameters()) == 67225,
       "head 17 -> 16 -> 1, 67,225 parameters per seed")
f = torch.randn(3, 16)                       # probe values for the 16 pooled code statistics (not data)
e = torch.tensor([[0.0], [0.2], [0.8]])      # probe values for phi_u (not data)
mu, sd = float(d.emu[0]), float(d.esd[0])
report("active: x17 = standardised phi_u", all(torch.allclose(a.head_in(f, e)[:, 16], (e[:, 0] - mu) / sd) for a in act), f"mu_phi={mu:.6f}, sigma_phi={sd:.6f}")
report("neutralized: x17 = 0", all(torch.equal(n.head_in(f, e)[:, 16], torch.zeros(3)) for n in neu))
report("16 learned inputs identical in both options", all(torch.equal(a.head_in(f, e)[:, :16], n.head_in(f, e)[:, :16]) for a, n in zip(act, neu)))
print(f"frozen ensemble threshold tau = {spec['decision']['threshold']}")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
