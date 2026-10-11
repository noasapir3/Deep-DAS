# Submission status

## Final
- **Report.** The final report is v3 ([report/DeepDAS_report_v3.pdf](report/DeepDAS_report_v3.pdf)). It was approved by the authors on 2026-10-11, including the *Team contributions* paragraph. The previous version, v2.2, is archived in [report/archive/v2_2/](report/archive/v2_2/).
- **Contributions.** The same approved wording is on the front page ([README.md](README.md#status-and-credits)).
- **Code link.** https://github.com/noasapir3/Deep-DAS (processed data: release `data-v1.0`).

## Publication

| Item | State |
|---|---|
| Branch | The approved version is on the branch `submission`. |
| Commit authorship | The commits on `submission` are authored by Noa Sapir. The first five are copies, with the same content, of the commits on `main`; the approved report and documentation changes follow in one commit. |
| The original record | `main` is kept unchanged, with its original commits (made under the identity of the AI assistant used in the workspace). The tag `data-v1.0` and its data release are unchanged. |
| Use of software tools | Stated on the front page ([README.md](README.md#status-and-credits)). |

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
