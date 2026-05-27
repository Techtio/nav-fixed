#!/usr/bin/env python3
"""Host-side batch runner for peak validator. Runs each seed as isolated docker exec."""
import subprocess, time, json, csv, os
from datetime import datetime

SEEDS = [0, 7, 13, 31, 42]
C = "kuavo_clean_craic"
N_ATT = 3
NOW = datetime.now().strftime("%Y-%m-%d_%H%M")
DIR = os.path.expanduser(f"~/Desktop/test_report/{NOW}")
os.makedirs(DIR, exist_ok=True)

COLS = ['seed','attempt','candidate_id','source','x','y','yaw_deg','orig_score',
        'right_peak','right_peak_err','right_count','right_med','right_p90','right_span','right_ok',
        'back_peak','back_peak_err','back_count','back_med','back_p90','back_span','back_ok',
        'fix_score','selected','reject_reason','gt_ok','gt_x','gt_y','gt_yaw_deg','pos_err','yaw_err_deg']

all_rows = []
summary = {}
total_attempts = 0
candidate_coverage = 0
validator_total = 0
validator_correct = 0

for seed in SEEDS:
    print(f"\n{'─'*50}\nSEED {seed}\n{'─'*50}")
    subprocess.run(f"docker exec {C} pkill -9 -f ros 2>/dev/null", shell=True, capture_output=True)
    subprocess.run(f"docker exec {C} pkill -9 -f python 2>/dev/null", shell=True, capture_output=True)
    time.sleep(5)
    subprocess.run(f"docker exec {C} rm -f /tmp/nav_test/logs/fv_seed{seed}.json", shell=True, capture_output=True)

    subprocess.run(f"docker exec -d -e DISPLAY=:0 {C} bash -c \""
                   f"nohup bash -c 'source /opt/ros/noetic/setup.bash; source /root/kuavo_ws/devel/setup.bash; "
                   f"export LD_LIBRARY_PATH=/opt/drake/lib:/root/kuavo_ws/installed/lib:\\$LD_LIBRARY_PATH; "
                   f"export ROBOT_VERSION=52 DISPLAY=:0 PYTHONUNBUFFERED=1; cd /tmp/nav_test; "
                   f"rm -f __pycache__/fixed_pose_validator*.pyc; "
                   f"python3 -u fixed_pose_validator.py --seeds {seed} --attempts {N_ATT}' "
                   f"> /tmp/nav_test/logs/fv_peak_s{seed}.log 2>&1 &\"", shell=True, capture_output=True)

    # Wait for results (check log for "SUMMARY" marker)
    for i in range(36):
        time.sleep(10)
        r = subprocess.run(f"docker exec {C} grep -c 'SUMMARY: DOMINANT PEAK' /tmp/nav_test/logs/fv_peak_s{seed}.log 2>/dev/null",
                          shell=True, capture_output=True, text=True)
        if int(r.stdout.strip() or "0") > 0: break

    # Extract log lines
    log = subprocess.run(f"docker exec {C} cat /tmp/nav_test/logs/fv_peak_s{seed}.log",
                        shell=True, capture_output=True, text=True)
    for line in log.stdout.split('\n'):
        if any(x in line for x in ['CASE1', 'CASE2', 'CASE3', 'SEED', 'GT:', 'SELECTED', 'BEST_GT', 'candidate_coverage', 'selected_pass']):
            print(line)

    # Read JSONL
    from subprocess import check_output
    jl = subprocess.run(f"docker exec {C} cat /tmp/nav_test/logs/peak_validator.jsonl 2>/dev/null",
                       shell=True, capture_output=True, text=True)
    seed_rows = []
    for line in jl.stdout.strip().split('\n'):
        if line.strip():
            try: seed_rows.append(json.loads(line))
            except: pass

    if seed_rows:
        all_rows.extend(seed_rows)
        for a in range(1, N_ATT+1):
            ar = [r for r in seed_rows if r['attempt']==a]
            if not ar: continue
            gt_ok = [r for r in ar if r.get('gt_ok')]
            sel = [r for r in ar if r.get('selected')]
            n_gt_ok = len(gt_ok)
            sel_pass = sel and sel[0].get('gt_ok')

            total_attempts += 1
            if n_gt_ok > 0:
                candidate_coverage += 1
                validator_total += 1
                if sel_pass: validator_correct += 1

            case = "CASE1_NO_GT" if n_gt_ok==0 else ("CASE3_PASS" if sel_pass else "CASE2_MISRANK")
            summary.setdefault(seed, {})[a] = {"case": case, "pass": sel_pass, "n_gt_ok": n_gt_ok}
    else:
        print(f"  NO ROWS")
        for a in range(1, N_ATT+1):
            summary.setdefault(seed, {})[a] = {"case": "SKIP", "pass": False, "n_gt_ok": 0}

# ═══ Write CSV ═══
csv_path = os.path.join(DIR, "peak_validator_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader(); w.writerows(all_rows)
jl_path = os.path.join(DIR, "peak_validator_results.jsonl")
with open(jl_path, "w") as f:
    for r in all_rows: f.write(json.dumps(r) + "\n")

# ═══ Summary ═══
print("\n" + "=" * 70)
print("PEAK VALIDATOR BATCH SUMMARY")
print("=" * 70)
print(f"{'seed':<6} {'A1':<14} {'A2':<14} {'A3':<14} {'gt_ok_cov':<12}")
print("-" * 62)
for s in SEEDS:
    sd = summary.get(s, {})
    astr = []; gt = 0
    for a in [1,2,3]:
        d = sd.get(a, {"case":"SKIP","pass":False,"n_gt_ok":0})
        mark = "✓" if d["pass"] else "✗"
        astr.append(f"{d['case'][:6]}:{mark}")
        if d["n_gt_ok"] > 0: gt += 1
    print(f"{s:<6} {astr[0]:<14} {astr[1] if len(astr)>1 else 'N/A':<14} "
          f"{astr[2] if len(astr)>2 else 'N/A':<14} {gt}/3")

print("-" * 62)
va = validator_correct / max(validator_total, 1)
sp = sum(1 for sd in summary.values() for d in sd.values() if d["pass"])
print(f"candidate_coverage:   {candidate_coverage}/{total_attempts or 1}")
print(f"validator_accuracy:   {validator_correct}/{validator_total or 1} = {va:.0%}")
print(f"selected_pass:        {sp}/{total_attempts or 1}")
print(f"Target: cov≥13  acc≥90%  pass≥13")
print(f"Status: {'✅' if candidate_coverage>=13 and va>=0.9 and sp>=13 else '❌'}")
print(f"\nData: {DIR}/")

# Copy logs
for s in SEEDS:
    subprocess.run(f"docker cp {C}:/tmp/nav_test/logs/fv_peak_s{s}.log {DIR}/ 2>/dev/null", shell=True)
    subprocess.run(f"docker cp {C}:/tmp/nav_test/logs/peak_validator.jsonl {DIR}/peak_validator.jsonl 2>/dev/null", shell=True)
    subprocess.run(f"docker cp {C}:/tmp/nav_test/logs/peak_validator.csv {DIR}/peak_validator.csv 2>/dev/null", shell=True)
print("Done.")
