"""Shared handling of --phi-mode for the reproduction scripts (v1.1 review copy). See src/deepdas/ensemble/phi_mode.py."""
import json, sys
from pathlib import Path
import pandas as pd
from deepdas.ensemble import phi_mode

sys.argv = phi_mode.parse_cli(sys.argv)          # strips --phi-mode X and exports DEEPDAS_PHI_MODE for worker processes
MODE = phi_mode.current()


def default_out(paths, sub: str) -> Path:
    return (paths.RESULTS / sub) if MODE == "active" else (paths.ROOT / "results_phi_neutralized" / sub)


def stamp(out: Path) -> None:
    """Record the option in the output folder. In 'neutralized' mode also add a phi_mode column to every CSV and a key to every JSON
    (active-mode tables are left byte-identical to the published ones; their folder carries RUN_MODE.json)."""
    out = Path(out); (out / "RUN_MODE.json").write_text(json.dumps(phi_mode.describe(MODE), indent=1))
    if MODE == "active":
        return
    for f in out.rglob("*.csv"):
        d = pd.read_csv(f)
        if "phi_mode" not in d.columns:
            d.insert(0, "phi_mode", MODE); d.to_csv(f, index=False)
    for f in out.rglob("*.md"):
        t = f.read_text()
        if not t.startswith("> **phi_mode"):
            f.write_text("> **phi_mode = neutralized** - inference-time intervention on the frozen hybrid model (standardised phi_u input held at 0). "
                         "Post-hoc analysis; not a model trained without phi_u; not an independent evaluation. 'DeepDAS Ensemble v1' below means "
                         "this neutralized configuration.\n\n" + t)
    for f in out.rglob("*.json"):
        if f.name == "RUN_MODE.json":
            continue
        j = json.loads(f.read_text())
        if isinstance(j, dict) and "phi_mode" not in j:
            j = {"phi_mode": MODE, **j}; f.write_text(json.dumps(j, indent=1))
