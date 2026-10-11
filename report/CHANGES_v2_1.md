# Report v2.1: precision fixes made to a copy of report v2

Report v2 (`../v2/`) is unchanged. The structure, figures 2–4, the tables, the results and the status of the main model are the same as in v2. The report is still 5 pages.

| # | Fix | Where |
|---|---|---|
| 1 | The architecture figure (`figs/G0_architecture_two_options`) shows both options for the 17th head input:<br>- `active`: the standardised φ_u, the model as trained and tested, used for all main results;<br>- `neutralized`: x₁₇ = 0, the post-hoc check.<br>The script asserts both options against the frozen checkpoints and asserts that the 16 learned inputs are identical (`figure_code/arch_figure_two_options.py`). | Fig. 1 and its caption |
| 2 | "No truth channel is an input" now refers to the deep networks only. The Z-score's 13 truth-error features are named. | Fig. 1 caption; Discussion, "Coverage and realism" |
| 3 | Coverage: "a third" is replaced by "700 of 1,708 normal vehicles (41 %)". The figure is 1,708 − 1,008 decided normals, from `T2_overall.csv`. | Discussion, "Coverage and realism" |
| 4 | Hard braking: "missed by all" is replaced by "weak for all detectors (at most 1/11)". The counts are 1/11, 0/11 and 0/12 (`T1_per_type.csv`). | Fig. 4 caption |
| 5 | Reproduction claim: final-model and baseline results are reproduced by the release; stage-1 and study numbers come from archived records. | Abstract; Reproducibility |

**Wording tightened to keep 5 pages:**
- one clause in the abstract;
- the Klausner sentence (same content);
- the sentence about the v1.1 review copy;
- limitation (7);
- Figure 1 moved earlier in the source, with the same placement on page 2.

**Build checks:** 5 pages; no unresolved references; 28 fonts embedded.
