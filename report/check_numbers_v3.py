#!/usr/bin/env python3
"""Number check for report v3 (a rewrite of report v2.2 that must not introduce unsourced numbers).

1. Every number in the body of report_v3.tex must also occur in report/archive/v2_2/report_v2_2.tex (whose 35 quantitative statements are
   mapped to their sources by research/08_tailgating_feature_intervention/code/number_sources.py), or
2. be listed in NEW below with the repository file and the pattern that must be found in it.
Usage: python report/check_numbers_v3.py   (from the repository root). Exit code 1 if any number is unsourced.
"""
import re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3, V22 = ROOT / "report/report_v3.tex", ROOT / "report/archive/v2_2/report_v2_2.tex"


def body(p):
    t = p.read_text()
    t = t[t.index("\\begin{document}"):t.index("\\begin{thebibliography}")]
    t = re.sub(r"\\(label|ref|cite|includegraphics|graphicspath)(\[[^]]*\])?\{[^}]*\}", " ", t)
    t = t.replace("{,}", "").replace("\\,", " ")
    return re.sub(r"\\[A-Za-z]+", " ", t)          # drop LaTeX command names such as \\ge before 0.28


def numbers(t):
    return set(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?:/\d+)?", t))


T1 = pd.read_csv(ROOT / "results/final_test/T1_per_type.csv").set_index("type")
NEW = {  # number in v3 -> (meaning, source file, regex or callable on the file text)
    "8/12": ("single model, lane straddling, final test (common subset)", "results/final_test/T1_per_type.csv",
             lambda t: int(T1.loc["lane_straddling", "CR_P99_detected"]) == 8 and int(T1.loc["lane_straddling", "common_n"]) == 12),
    "9/10": ("single model, collision, final test (common subset)", "results/final_test/T1_per_type.csv",
             lambda t: int(T1.loc["collision", "CR_P99_detected"]) == 9 and int(T1.loc["collision", "common_n"]) == 10),
    "10/10": ("Z-score, collision, final test (common subset)", "results/final_test/T1_per_type.csv",
              lambda t: int(T1.loc["collision", "Z_detected"]) == 10 and int(T1.loc["collision", "common_n"]) == 10),
    "18": ("validation stalls in the stop-mode comparison (Fig. 2b)", "research/04_stall_diversity/README.md", r"18 stalls \(6 per mode"),
    "632": ("normal vehicles in the stage-1 comparison", "research/01_unsupervised_autoencoder/README.md", r"91 injected vehicles vs 632 held-out normal vehicles"),
    "0.96": ("overweight: encoder probe AUC", "research/06_weak_type_diagnosis/README.md", r"\| Overweight \| 0\.87 \| 0\.96 \| \*\*0\.83\*\* \|"),
    "0.83": ("overweight: ensemble score AUC", "research/06_weak_type_diagnosis/README.md", r"\| Overweight \| 0\.87 \| 0\.96 \| \*\*0\.83\*\* \|"),
    "5": ("latent score: 5 of the 9 original scenarios (and other small counts already in v2.2)", "research/01_unsupervised_autoencoder/records/unsup_release_primary_latent_maxZ_quality_P99_RELEASE_REPORT.md",
          r"Causing vehicles detected: \*\*5 of 9\*\*"),
}
STRUCTURAL = {"1", "2", "3", "4", "6", "7", "16", "17"}   # section numbers, Q1-Q4, Fig./Table numbers, 16 learned + 1 = 17 head inputs


def statement_checks():
    """Statements of v3 whose wording changed although their numbers already occur in v2.2."""
    X = pd.read_csv(ROOT / "results/final_test/vehicle_outcomes.csv")
    c = X[(X.block == "tg_control") & (X.label == "normal_matched_control") & X.common]
    got = tuple(f"{int(c[f'{d}_alarm'].sum())}/{len(c)}" for d in ("E_DIV", "CR_P99", "Z"))
    ok = got == ("7/11", "0/11", "0/11") and "tailgating matched controls 7/11, 0/11 and 0/11" in V3.read_text()
    print(f"tailgating matched controls, common subset (E / single / Z) = {got} -> {'match' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def main():
    old, new = numbers(body(V22)), numbers(body(V3))
    rows, bad = [], statement_checks()
    for n in sorted(new - old, key=lambda s: (len(s), s)):
        if n in NEW:
            meaning, f, rx = NEW[n]; txt = (ROOT / f).read_text(errors="ignore")
            ok = rx(txt) if callable(rx) else bool(re.search(rx, txt))
            rows.append((n, meaning, f, "match" if ok else "NOT FOUND")); bad += not ok
        elif n in STRUCTURAL:
            rows.append((n, "structural (section / question / figure number)", "-", "n/a"))
        else:
            rows.append((n, "UNSOURCED", "-", "MISSING")); bad += 1
    for r in rows:
        print(" | ".join(r))
    print(f"{len(new)} distinct numbers in v3; {len(new & old)} also in v2.2; {len(new - old)} new; {bad} unsourced")
    gone = sorted(old - new, key=lambda s: (len(s), s))
    print("numbers of v2.2 not repeated in v3 (moved to the documentation or dropped as repetition):", ", ".join(gone))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
