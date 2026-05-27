#!/usr/bin/env python3
"""Simple param sweep — each (threshold, seed) is a nohup docker exec. Reads JSON after completion."""
import subprocess, time, json, csv, os

THRS = [0.05, 0.07, 0.10]
SEEDS = [0, 7, 13, 31, 42]
C = "kuavo_clean_craic"
N_ATT = 3
REPORT = os.path.expanduser("~/Desktop/test_report")
os.makedirs(REPORT, exist_ok=True)

COLS = ['threshold','seed','attempt','candidate_id','source','x','y','yaw_deg',
        'orig_score','right_count','right_med','right_p90','right_span',
        'right_density','right_ok','back_count','back_med','back_p90',
        'back_span','back_density','back_ok','fix_score','selected',
        'reject_reason','gt_x','gt_y','gt_yaw_deg','pos_err','yaw_err_deg','pass']

all_rows = []
summary = {}

for thr in THRS:
    print(f"\n{'='*50}\nTHRESHOLD = {thr}\n{'='*50}")
    summary[thr] = {}
    for seed in SEEDS:
        print(f"  seed={seed} ...", end=" ", flush=True)
        
        # Clean old ROS
        subprocess.run(f"docker exec {C} pkill -9 -f ros 2>/dev/null", shell=True, capture_output=True)
        subprocess.run(f"docker exec {C} pkill -9 -f python 2>/dev/null", shell=True, capture_output=True)
        time.sleep(5)
        
        # Launch background test
        subprocess.run(f"docker exec -d -e DISPLAY=:0 {C} bash -c \""
                       f"nohup bash -c '"
                       f"source /opt/ros/noetic/setup.bash; "
                       f"source /root/kuavo_ws/devel/setup.bash; "
                       f"export LD_LIBRARY_PATH=/opt/drake/lib:/root/kuavo_ws/installed/lib:\\$LD_LIBRARY_PATH; "
                       f"export ROBOT_VERSION=52 DISPLAY=:0 PYTHONUNBUFFERED=1; "
                       f"cd /tmp/nav_test; "
                       f"python3 -u fv_sweep_seed.py {seed} {thr} "
                       f"' &\"", shell=True, capture_output=True)
        
        key = f"t{thr}_s{seed}"
        json_path = f"/tmp/nav_test/logs/swp_t{str(thr).replace('.','p')}_s{seed}.json"
        
        # Wait for JSON file to appear (poll up to 10 min)
        ok = False
        for i in range(60):
            time.sleep(10)
            check = subprocess.run(f"docker exec {C} cat {json_path} 2>/dev/null | wc -c",
                                   shell=True, capture_output=True, text=True)
            if int(check.stdout.strip() or "0") > 10:
                ok = True; break
        
        if ok:
            cp = subprocess.run(f"docker cp {C}:{json_path} /tmp/{key}.json", shell=True, capture_output=True)
            try:
                rows = json.load(open(f"/tmp/{key}.json"))
                for r in rows: r["threshold"] = thr
                all_rows.extend(rows)
                
                att_results = []
                for a in range(1, N_ATT+1):
                    ar = [r for r in rows if r["attempt"]==a]
                    sel = [r for r in ar if r["selected"]]
                    if not ar: att_results.append("SKIP")
                    elif any(r["pass"] for r in sel): att_results.append("PASS")
                    elif sel: att_results.append("SEL")
                    else: att_results.append("NO_SEL")
                summary[thr][seed] = att_results
                pc = att_results.count("PASS")
                print(f"OK {len(rows)} rows {att_results} PASS={pc}/{N_ATT}")
            except Exception as e:
                summary[thr][seed] = ["ERR"]*N_ATT
                print(f"JSON_ERR: {e}")
        else:
            summary[thr][seed] = ["TIMEOUT"]*N_ATT
            print("TIMEOUT")

# ═══ Full CSV ═══
csv_path = os.path.join(REPORT, "param_sweep_full.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader(); w.writerows(all_rows)
jsonl_path = os.path.join(REPORT, "param_sweep_full.jsonl")
with open(jsonl_path, "w") as f:
    for row in all_rows: f.write(json.dumps(row) + "\n")

# ═══ Summary ═══
print("\n" + "=" * 70)
print("SWEEP SUMMARY")
print("=" * 70)
for thr in THRS:
    sd = summary[thr]
    total_p = sum(sd[s].count("PASS") for s in SEEDS)
    no_sel = sum(sd[s].count("NO_SEL") for s in SEEDS)
    crash = sum(sd[s].count("TIMEOUT") + sd[s].count("ERR") for s in SEEDS)
    total_t = sum(len([a for a in sd[s] if a not in ("SKIP","TIMEOUT","ERR")]) for s in SEEDS)
    swap = 0
    for s in SEEDS:
        srows = [r for r in all_rows if r["threshold"]==thr and r["seed"]==s]
        swap += sum(1 for r in srows if r["selected"] and not r["pass"])
    print(f"\nTHR={thr:.2f}: {total_p}/{total_t} PASS  NO_SEL={no_sel} WRONG={swap} CRASH={crash}  "
          f"{'✅' if total_p>=13 else '❌'}")
    print(f"{'seed':<6} {'att1':<8} {'att2':<8} {'att3':<8} pass")
    for s in SEEDS:
        a = sd.get(s, ["???"]*N_ATT)
        print(f"{s:<6} {a[0]:<8} {a[1] if len(a)>1 else 'N/A':<8} "
              f"{a[2] if len(a)>2 else 'N/A':<8} {a.count('PASS')}/{N_ATT}")

print(f"\n{jsonl_path} ({len(all_rows)} rows)")
