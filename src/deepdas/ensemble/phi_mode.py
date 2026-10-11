"""
Inference option for the explicit tailgating summary feature phi_u (v1.1 review copy).

DeepDAS Ensemble v1 was TRAINED with phi_u as the 17th input of its classifier head (all five checkpoints, feats = ['tg']).
Two documented inference options exist:

  active       (default) the original hybrid model exactly as trained, selected and tested:
               head input 17 = (phi_u - emu) / esd, emu = 0.005994, esd = 0.031635 (stored in every checkpoint)
  neutralized  INFERENCE-TIME INTERVENTION on the same frozen weights: head input 17 := 0 in the standardised coordinate,
               i.e. the value of an average training-normal vehicle. The 16 learned pooled features are unchanged.
               This measures how much the frozen detector depends on phi_u; it is NOT a model trained without phi_u, it was
               defined after the final test had been exposed (post hoc), and it is not an independent evaluation.

Select with  --phi-mode active|neutralized  on the scripts, or the environment variable DEEPDAS_PHI_MODE.
In 'neutralized' mode every evaluation uses the FROZEN per-seed statistics (mu_k, sigma_k) and the FROZEN ensemble threshold tau of
models/deepdas_ensemble_v1/ensemble.json; nothing is re-standardised or re-calibrated.
"""
from __future__ import annotations

import os

MODES = ("active", "neutralized")
ENV = "DEEPDAS_PHI_MODE"


def current() -> str:
    m = os.environ.get(ENV, "active")
    if m not in MODES:
        raise ValueError(f"{ENV} must be one of {MODES}, got {m!r}")
    return m


def describe(mode: str | None = None) -> dict:
    mode = mode or current()
    return {"phi_mode": mode,
            "model": "DeepDAS Ensemble v1 (E_stall_diversity_ENSEMBLE), five frozen checkpoints trained WITH phi_u (head 17->16->1)",
            "phi_input": "standardised phi_u (as trained)" if mode == "active" else "standardised phi_u input held at 0 (inference-time intervention)",
            "ensemble_statistics": "frozen mu_k, sigma_k and tau from ensemble.json",
            "status": "original hybrid model as selected and tested" if mode == "active" else
                      "post-hoc dependency analysis on exposed data; not a model trained without phi_u; not an independent evaluation"}


def parse_cli(argv: list) -> list:
    """Remove '--phi-mode X' from argv, export it to the environment (inherited by worker processes) and return the cleaned argv."""
    if "--phi-mode" in argv:
        i = argv.index("--phi-mode"); mode = argv[i + 1]
        if mode not in MODES:
            raise SystemExit(f"--phi-mode must be one of {MODES}")
        os.environ[ENV] = mode; argv = argv[:i] + argv[i + 2:]
    return argv
