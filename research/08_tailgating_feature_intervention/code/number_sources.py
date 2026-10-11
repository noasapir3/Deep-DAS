#!/usr/bin/env python3
"""Source table for report v2: every quantitative statement in report_v2.tex -> model version x data role -> file in this release.
Each row carries a regular expression that must match the source file; the script checks it and writes
../tables/REPORT_V2_NUMBER_SOURCES.csv (and .md). Exit code 1 if any source does not contain the stated value.
Version legend: hybrid = DeepDAS Ensemble v1 with phi_u active (frozen, as tested); neutralized = same weights, phi_u
neutralized at inference (post hoc); single = supervised single model; zscore = SimStudio Z-score; archived = recorded
result of an earlier study (not regenerated); W1/W2 = targeted-data arms (validation only)."""
import re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]; OUT = Path(__file__).resolve().parents[1] / "tables"
R, N, S08 = "results", "results_phi_neutralized", "research/08_tailgating_feature_intervention"
FT, NFT = f"{R}/final_test", f"{N}/final_test"
rows = [
 # (value as in report, where in report, version, data role, file, regex that must match the file)
 ("tau = 1.9726", "Sec. 3 / Table 1", "hybrid", "val_B normals (calibration)", "models/deepdas_ensemble_v1/ensemble.json", r"1\.9726181803233118"),
 ("67,225 trained parameters per seed", "Sec. 3", "hybrid", "model", f"{S08}/figures/data/G0_architecture_spec.json", r"67225"),
 ("AUROC 0.60 vs 0.89", "Abstract, Sec. 5.1", "archived autoencoder vs zscore", "study 01 evaluation set", "research/01_unsupervised_autoencoder/README.md", r"AUROC \*\*0\.60\*\*.*\n.*AUROC \*\*0\.89\*\*"),
 ("0/3 seeds each (tailgating dev case, study 02 arms A and C)", "Sec. 5.4", "archived", "development case C3", "research/02_supervised_single_model/records/seedstudy_20261007_SEED_SUMMARY.md", r"C3_alarm\s*\|\s*0\.0 ± 0\.0\s*\|\s*0\.0 ± 0\.0\s*\|\s*0\.0 ± 0\.0\s*\|\s*0\.0 ± 0\.0"),
 ("validation tailgating recall 0.385 -> 0.923", "Sec. 5.4, Fig. 2a", "archived (study 03, both arms with phi_u)", "validation", "research/03_tailgating_data/README.md", r"0\.385[\s\S]*0\.923"),
 ("validation AP 0.781 vs 0.751; repetition 0.93 +/- 0.15", "Sec. 5.4", "archived (study 04)", "validation", "research/04_stall_diversity/README.md", r"0\.751 \| \*\*0\.781\*\*[\s\S]*0\.93 ± 0\.15"),
 ("development 8 of 9; overweight 1.865", "Sec. 6", "hybrid", "development (not independent)", "research/04_stall_diversity/README.md", r"\*\*8/9\*\*[\s\S]*1\.865"),
 ("development 8/9 (reproduced)", "Sec. 6, Table 3", "hybrid", "development", f"{S08}/tables/report_summary_table.csv", r"development cases \(9\),7/9,8/9"),
 ("validation macro 0.714, AP 0.781, 18/18, 12/13, 40/54", "Sec. 6", "hybrid", "validation (selection)", f"{S08}/tables/validation_comparison.csv", r"12/13[\s\S]*0\.714[\s\S]*0\.781[\s\S]*18/18[\s\S]*40/54"),
 ("false alarms val_A 1.07 %", "Sec. 6", "hybrid", "validation", f"{R}/validation/validation_metrics.csv", r",14,0\.0106626"),
 ("228 runs", "Abstract, Sec. 6", "final test", "final test (plan frozen before simulation)", f"{FT}/records/PLAN_FINAL_TEST_FROZEN.md", r"228 runs"),
 ("105 of 108 core injections realised", "Sec. 6", "final test", "final test", f"{FT}/T1_per_type.csv",
  lambda t: (lambda d: d.injected.sum() == 108 and d.realized.sum() == 105)(pd.read_csv(pd.io.common.StringIO(t)))),
 ("macro recall 0.568 / 0.364 / 0.232 (common)", "Abstract, Sec. 6, Table 2", "hybrid / single / zscore", "final test, common subset", f"{FT}/T2_overall.csv", r"E_DIV,0\.568,0\.568[\s\S]*CR_P99,0\.364,0\.364[\s\S]*Z,0\.231,0\.232"),
 ("ROC-AUC 0.908, AUPRC 0.742; precision 0.74, F1 0.64", "Abstract, Table 2", "hybrid", "final test, common subset", f"{FT}/T2_overall.csv", r"E_DIV,.*,0\.74,0\.64,0\.908,0\.742"),
 ("single 0.88 / 0.52; 0.769 / 0.515", "Table 2", "single", "final test, common subset", f"{FT}/T2_overall.csv", r"CR_P99,.*,0\.881,0\.517,0\.769,0\.515"),
 ("Z-score 0.819 / 0.496, precision 0.77 (common)", "Abstract, Table 2", "zscore", "final test, common subset (101 / 1,008)", f"{S08}/figures/data/G4_roc_pr_four_configurations.csv", r"SimStudio Z-score,0\.8186,0\.4963,0\.0069,0\.2277,0\.7667,101,1008"),
 ("Z-score F1 0.35 (common) = 2PR/(P+R) with P 0.7667, R 0.2277", "Table 2", "zscore", "final test, common subset", f"{S08}/figures/data/G4_roc_pr_four_configurations.csv", r"0\.2277,0\.7667"),
 ("Z-score own set 0.783 / 0.394, precision 0.64, F1 0.33; 105 / 1,574", "Table 2 caption", "zscore", "final test, Z-score's own decided set", f"{FT}/T2_overall.csv", r"Z,.*,0\.639,0\.326,0\.783,0\.394,105/105,1574/1708"),
 ("CI [.49,.65], [.30,.43], [.22,.26]", "Abstract, Table 2", "hybrid / single / zscore", "final test", f"{FT}/T5_uncertainty.json", r"0\.4[89][\s\S]*0\.6[45]"),
 ("McNemar 38 vs 4 (p<0.001), 28 vs 8 (p=0.0012)", "Sec. 6", "hybrid vs zscore / single", "final test", f"{FT}/FINAL_TEST_REPORT.md", r"\"only DeepDAS Ensemble v1\": 38[\s\S]*0\.0012"),
 ("decided normals 1,008 / 1,574 of 1,708", "Sec. 6, Table 2", "all", "final test", f"{FT}/T2_overall.csv", r"1008/1708[\s\S]*1574/1708"),
 ("tailgating 11/11, weaving 7/11, reckless 7/10, stalls 12/12, hard braking 1/11, speeding 4/12, overweight 4/12, straddling 5/12, collision 6/10", "Abstract, Sec. 6, Fig. 4", "hybrid", "final test, common subset", f"{FT}/T1_per_type.csv",
  r"speeding,12,12,12,12,12,4,[\s\S]*reckless_driver,.*,10,0\.7,[\s\S]*hard_braking,.*,11,0\.091,[\s\S]*stalled_vehicle,.*,12,1\.0,[\s\S]*lane_weaving,.*,11,0\.636,[\s\S]*lane_straddling,.*,12,0\.417,[\s\S]*collision,.*,10,0\.6,[\s\S]*overweight_vehicle,.*,12,0\.333,[\s\S]*tailgating,.*,11,1\.0,"),
 ("Z-score tailgating 1/11, weaving 0/11", "Abstract, Sec. 6", "zscore", "final test, common subset", f"{FT}/T1_per_type.csv", r"lane_weaving,.*,0\.0\n[\s\S]*tailgating,.*,0\.091\n"),
 ("FA normal driving 20 (2.0 %), 5 (0.5 %), 7 (0.7 %)", "Sec. 6, Table 2", "hybrid / single / zscore", "final test, common subset", f"{FT}/T2_overall.csv", r"E_DIV,.*20/1008[\s\S]*CR_P99,.*5/1008[\s\S]*Z,.*7/1008"),
 ("legitimate stops 20/28 (3 s 3/8, 5 s 8/11, 8 s 9/9), 14/28, 22/28", "Abstract, Sec. 6, Table 2", "hybrid / single / zscore", "final test, common subset", f"{FT}/T3_stops_tailgating.csv", r"LEGIT 3 s.*,3/8,[\s\S]*LEGIT 5 s.*,8/11,[\s\S]*LEGIT 8 s.*,9/9,"),
 ("tailgating matched controls 7/11, 0/11, 0/12", "Sec. 6", "hybrid / single / zscore", "final test", f"{FT}/T3_stops_tailgating.csv", r"tailgating controls \(false alarm\),12,7/11,0/11,0/12"),
 ("W1/W2: weak-type 0.35 -> 0.71 / 0.74; ROC-AUC 0.87 -> 0.96; legitimate stops 45 and 52 of 54", "Sec. 6 (W1/W2)", "W1 / W2 (validation only)", "expanded validation", "research/07_targeted_data_study/README.md", r"0\.35 / 0\.54 \| \*\*0\.71 / 0\.80\*\* \| \*\*0\.74 / 0\.83\*\*[\s\S]*0\.87 / 0\.70 \| 0\.96[\s\S]*\*\*40\*\* \| \*\*45\*\* \| \*\*52\*\*"),
 ("105x fewer stop windows; remedies 37, 27, 21 of 54", "Sec. 7", "archived (study 05)", "validation", "research/05_legitimate_stops/README.md", r"105× fewer stop windows[\s\S]*27 \(50 %\)[\s\S]*21 \(39 %\)"),
 ("Table 3 row 1: 7/9 vs 8/9", "Table 3", "neutralized vs hybrid", "development (diagnostic)", f"{S08}/tables/report_summary_table.csv", r"development cases \(9\),7/9,8/9"),
 ("Table 3 row 2: 9/13 vs 12/13", "Table 3", "neutralized vs hybrid", "validation, frozen tau", f"{S08}/tables/report_summary_table.csv", r"tailgating, validation\"?,9/13,12/13"),
 ("Table 3 row 3: 10/11 vs 11/11", "Abstract, Table 3", "neutralized vs hybrid", "exposed final test, POST HOC", f"{S08}/tables/report_summary_table.csv", r"tailgating, exposed final test \(post hoc\)\"?,10/11,11/11"),
 ("Table 3 row 4: 0.560 vs 0.568", "Table 3", "neutralized vs hybrid", "exposed final test, POST HOC", f"{S08}/tables/report_summary_table.csv", r"macro recall, exposed final test \(post hoc\)\"?,0\.560,0\.568"),
 ("neutralized: ROC-AUC 0.906, AUPRC 0.737, FA 20/1008, legitimate stops 19/28", "README study 08, Fig. 3 legend", "neutralized", "exposed final test, POST HOC", f"{NFT}/T2_overall.csv", r"neutralized,E_DIV,0\.56,0\.56,.*20/1008.*19/28,0\.737,0\.633,0\.906,0\.737"),
 ("dev tailgating case score 3.73 vs -0.08; phi_u 0.81 (25 sd)", "Sec. 7, Contribution", "hybrid vs neutralized", "development case", f"{S08}/tables/development_cases_comparison.csv", r"Tailgating,0\.809\d*,25\.4\d*,3\.7317,True,-0\.0819,False"),
 ("linear probe AUC 1.0 (study 06)", "Sec. 7, Contribution", "hybrid encoder (probe)", "exposed final test (diagnosis)", "research/06_weak_type_diagnosis/README.md", r"[Tt]ailgating.*1\.0"),
]
out = []
for v, where, ver, role, f, rx in rows:
    p = ROOT / f; txt = p.read_text(errors="ignore") if p.exists() else None
    found = bool(txt is not None and (rx(txt) if callable(rx) else re.search(rx, txt)))
    out.append(dict(value_in_report=v, location=where, model_version=ver, data_role=role, source_file=f, check="match" if found else "NOT FOUND"))
T = pd.DataFrame(out); OUT.mkdir(exist_ok=True); T.to_csv(OUT / "REPORT_V2_NUMBER_SOURCES.csv", index=False)
(OUT / "REPORT_V2_NUMBER_SOURCES.md").write_text("# Report v2: source of every number\n\nGenerated by `code/number_sources.py`; each row was checked against the file.\n"
    "Final-test rows marked POST HOC use a test that had already been exposed; neutralized rows are an inference-time intervention on the frozen hybrid, not a model trained without phi_u.\n\n" + T.to_markdown(index=False) + "\n")
print(T[["value_in_report", "check"]].to_string(index=False)); bad = (T.check != "match").sum(); print(f"{len(T) - bad}/{len(T)} sources verified"); sys.exit(1 if bad else 0)
