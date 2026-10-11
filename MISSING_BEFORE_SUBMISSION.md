# Submission status

## Final
- **Report.** The final report is v3 ([report/DeepDAS_report_v3.pdf](report/DeepDAS_report_v3.pdf)). It was approved by the authors on 2026-10-11, including the *Team contributions* paragraph. The previous version, v2.2, is archived in [report/archive/v2_2/](report/archive/v2_2/).
- **Contributions.** The same approved wording is on the front page ([README.md](README.md#status-and-credits)).
- **Code link.** https://github.com/noasapir3/Deep-DAS (processed data: release `data-v1.0`).

## Still open

| Item | State |
|---|---|
| Publishing this documentation (branch `docs-v3`) | Not published. Planned order, after the authors approve the local package: a secure sign-in in the workspace, push `docs-v3`, then open a pull request for review. No direct update of `main`, and no force push. |
| Contributor attribution on GitHub | The published commits are attributed to the account `claude`. The supported options, and which of them would change history, are to be decided by the authors before publishing. |
| Course rules on disclosing the use of software tools | The front page states it in one sentence. The authors check it against the course guidelines. |

## The clean submission package
It is assembled outside the repository, in the authors' project folder, and contains:
- the final report PDF;
- its LaTeX source with figures;
- the repository link;
- a short README with checksums.

It does **not** contain:
- `_to_delete/`;
- transfer bundles (`*.bundle`) or transfer archives;
- backup and comparison copies (`baseline/`, `dl archive/`, earlier report versions, release backup archives);
- build logs, rendered previews or temporary files.

Nothing was deleted: the sources, the release folders and the full history stay in the project folders and in git.

## Decisions left open, not done (each would need explicit approval)
- **Word versions.** The Word versions of the report (`05_word/`, EN and HE) still follow report v1. They were not updated for v2 or v3.
- **Trained ablation.** No model without φ_u was trained on the final training data. Early single models without it exist (study 02, early data). `research/08_tailgating_feature_intervention/README.md` §6 lists what that claim would require.
