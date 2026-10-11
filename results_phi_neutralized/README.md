# Results with φ<sub>u</sub> neutralized (post-hoc analysis)

**What these files are.** The same evaluations as [`../results/`](../results/), produced by the same frozen weights with the 17th head input held at 0 (`--phi-mode neutralized`).

- **Frozen constants.** The per-seed statistics and τ are the frozen ones; nothing was re-estimated.
- **Stamping.** Every CSV has a `phi_mode` column, and every folder has a `RUN_MODE.json`.

**How to read them.**

- **Not a model trained without the feature.** These files measure how much the trained hybrid depends on φ<sub>u</sub>. They are not the performance of a model trained without it.
- **The final-test part is post hoc.** This option was defined after the final test had been exposed.
- **The reference is unchanged.** The published reference results remain those of the `active` hybrid in `../results/`.

| Folder | Content | Headline |
|---|---|---|
| `development/` | the 9 development cases | 7 / 9 (active: 8 / 9) |
| `validation/` | validation scores and metrics at the frozen τ | tailgating 9 / 13 (active: 12 / 13); macro recall 0.688 |
| `final_test/` | scores and tables T1–T7 (post hoc) | macro recall 0.560 (active: 0.568); ROC-AUC 0.906 |

**Regenerate and compare:**
```bash
python scripts/reproduce_final_test.py outputs/ft_neutralized --phi-mode neutralized
python scripts/verify_phi_modes.py
```

**Analysis:** [study 08](../research/08_tailgating_feature_intervention/) and [docs/INFERENCE_MODES.md](../docs/INFERENCE_MODES.md).
