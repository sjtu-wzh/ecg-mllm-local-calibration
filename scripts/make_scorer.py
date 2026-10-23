"""Create scripts/score_ecgbench.py from the official PULSE scorer (PULSE/evaluation/evaluate_ecgbench.py).
The official file is not redistributed here (the PULSE repository has no license file); this script applies the
minimal changes we used:
  1. placeholder paths (/path/to/PULSE/data/ECGBench, /path/to/eval_outputs) -> paths relative to this repository
  2. MMMU-ECG gold lookup uses item["id"] (the released ECGBench JSON has no "No" field)
  3. a comment marker missing in the official file ('auc = np.nan  If AUC ...' is a syntax error) is restored
Metric logic is unchanged. Run once after cloning https://github.com/AIMedLab/PULSE into ./PULSE:
    python scripts/make_scorer.py"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "PULSE/evaluation/evaluate_ecgbench.py")).read()
header = ('import os as _os\n'
          '_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root\n')
edits = [
    ('"/path/to/PULSE/data/ECGBench/', '_ROOT + "/data/ECGBench/'),
    ('"/path/to/eval_outputs"', '_ROOT + "/outputs/pulse7b"'),
    ('item["No"]', 'item["id"]'),
    ("auc = np.nan  If AUC cannot be calculated, NaN is returned", "auc = np.nan  # if AUC cannot be calculated, NaN is returned"),
]
for old, new in edits:
    if old not in src:
        raise SystemExit(f"expected text not found in the official scorer: {old!r} (has PULSE changed?)")
    src = src.replace(old, new)
out = os.path.join(ROOT, "scripts/score_ecgbench.py")
open(out, "w").write(header + src)
print("wrote", out)
