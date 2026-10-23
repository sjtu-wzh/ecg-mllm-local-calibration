"""Recompute the md5 hashes recorded in the three pre-registration freeze files and report any difference.
Files that are not shipped in the repository (official PULSE scorer, ECGBench-derived files, rendered images) are listed
with how to regenerate them. Run: python verify_freeze.py"""
import glob, hashlib, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
md5 = lambda p: hashlib.md5(open(os.path.join(ROOT, p), "rb").read()).hexdigest()
REGEN = {"scripts/score_ecgbench.py": "python scripts/make_scorer.py (from the official PULSE scorer)",
         "data/derived/ptbxl_cpsc9.json": "python scripts/build_derived_datasets.py (needs ECGBench)"}
EDITED_AFTER_RUN = {  # domain-analysis freeze: edited after the analysis had run; re-run reproduces the results exactly
    "clarity/label_prob.py": "outputs/clarity/domain_rerun_check.json", "scripts/score_ecgbench.py": "outputs/clarity/domain_rerun_check.json"}
FREEZES = [("SPH validation", "outputs/clarity/prereg_sph_freeze.json", lambda d: d["md5"]),
           ("Robustness experiment", "outputs/clarity/prereg_robust_freeze.json", lambda d: d["md5"]),
           ("Domain analysis", "outputs/clarity/preregistration_domain.json", lambda d: d["_freeze"]["md5"])]
for name, path, get in FREEZES:
    rec = json.load(open(os.path.join(ROOT, path)))
    print(f"\n{name}  ({path}; frozen at {rec.get('frozen_at') or rec['_freeze']['frozen_at']})")
    for f, h in get(rec).items():
        if not os.path.exists(os.path.join(ROOT, f)):
            print(f"  not shipped  {f}  -> {REGEN.get(f, 'see docs/REPRODUCE.md')}"); continue
        ok = md5(f) == h
        note = "" if ok or name != "Domain analysis" or f not in EDITED_AFTER_RUN else f"  (edited after the analysis ran; see {EDITED_AFTER_RUN[f]})"
        print(f"  {'OK        ' if ok else 'DIFFERENT '} {f}{note}")
rob = json.load(open(os.path.join(ROOT, "outputs/clarity/prereg_robust_freeze.json")))["perturbed_images"]
print("\nPerturbed images (regenerate with: python clarity/render_sph.py; python clarity/robust_setup.py)")
for cond, v in rob.items():
    fs = sorted(glob.glob(os.path.join(ROOT, f"data/sph_images_robust/{cond}/sph/*.png")))
    if not fs: print(f"  not present  {cond}"); continue
    h = hashlib.md5()
    for f in fs: h.update(hashlib.md5(open(f, "rb").read()).hexdigest().encode())
    print(f"  {'OK        ' if (len(fs), h.hexdigest()) == (v['n'], v['md5_of_md5s']) else 'DIFFERENT '} {cond} ({len(fs)} images)")
