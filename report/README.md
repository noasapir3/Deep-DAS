# Report

**The final report is [`DeepDAS_report_v3.pdf`](DeepDAS_report_v3.pdf)** (version 3, approved by the authors on 2026-10-11). It is 5 pages, including references, team contributions and the code link.

| File | Content |
|---|---|
| [`DeepDAS_report_v3.pdf`](DeepDAS_report_v3.pdf) | the final report |
| [`report_v3.tex`](report_v3.tex) and `figs_v3/` | its LaTeX source and figures; build with `pdflatex report_v3.tex`, run twice |
| [`CHANGES_v3.md`](CHANGES_v3.md) | what changed from v2.2, including the revision after review and the final changes (§8) |
| [`check_numbers_v3.py`](check_numbers_v3.py) | checks that every number in v3 is in v2.2 or in its source file |

**The report's structure:** problem → data and evaluation → the final model → four questions behind the design → results (the final test at the centre) → conclusions and limitations.

## Previous versions (archive)

| Version | Where | Status |
|---|---|---|
| v2.2 | [`archive/v2_2/`](archive/v2_2/): [PDF](archive/v2_2/DeepDAS_report_v2_2.pdf), [LaTeX](archive/v2_2/report_v2_2.tex), `figs/`, [changes from v2.1](archive/v2_2/CHANGES_v2_2.md), [precision fixes of v2.1](archive/v2_2/CHANGES_v2_1.md) | The version submitted before v3 (`report_final.pdf`). Kept unchanged; it still builds to the same text. |

Earlier versions (v1, v2, v2.1) are kept outside the repository, in the authors' project archive.

## Where the numbers come from
- **v2.2.** Every number is mapped to its model version, data role and source file in [REPORT_V2_NUMBER_SOURCES.md](../research/08_tailgating_feature_intervention/tables/REPORT_V2_NUMBER_SOURCES.md).
- **v3.** [`check_numbers_v3.py`](check_numbers_v3.py) confirms that every number in v3 is one of those or is found in its source file. Six numbers are new, and all are sourced.
- **Klausner.** The result comes from [the Klausner experiment](../models/deepdas_ensemble_v1/experiment_klausner/).

**Where each part of the report is documented in detail:** [docs/QUESTION_MAP.md](../docs/QUESTION_MAP.md).
