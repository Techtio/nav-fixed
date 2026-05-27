#!/usr/bin/env python3
"""
Parameter sweep for MAX_MED_RESIDUAL in fixed_pose_validator.
Tests thresholds 0.05, 0.07, 0.10 across 5 seeds × 3 attempts.
Host-side orchestrator — each seed runs as isolated docker exec.
"""
import subprocess, time, json, csv, os, sys

THRESHOLDS = [0.05, 0.07, 0.10]
SEEDS = [0, 7, 13, 31, 42]
C = "kuavo_clean_craic"
ATTEMPTS = 3
REPORT_DIR = os.path.expanduser("~/Desktop/test_report")
os.makedirs(REPORT_DIR, exist_ok=True)

COLS = ['threshold','seed','attempt','candidate_id','source','x','y','yaw_deg',
        'orig_score','right_count','right_med','right_p90','right_span',
        'right_density','right_ok','back_count','back_med','back_p90',
        'back_span','back_density','back_ok','fix_score','selected',
        'reject_reason','gt_x','gt_y','gt_yaw_deg','pos_err','yaw_err_deg','pass']

def run_seed(threshold, seed):
    """Run single seed test with given threshold. Returns list of result rows, launcher ref."""
    script = f'''
import rospy, sys, os, time, math, json, numpy as np
os.environ["DISPLAY"] = ":0"

# Override threshold BEFORE imports
import fixed_pose_validator as fv
fv.MAX_MED_RESIDUAL = {threshold}

sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/utils")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher
launcher = SimLauncher(scene="scene1", seed={seed}, robot_version=52)
launcher.start(node_name="swp_t{str(threshold).replace('.','p')}_s{seed}", timeout=120)
print("SIM_READY", flush=True)
time.sleep(8)

sys.path.insert(0, "/tmp/nav_test")
import corner_localizer as cl

gt = cl.GT_POSES.get({seed})

results = []
for att in range({ATTEMPTS}):
    time.sleep(1)
    xy, xy3, ang, ran = cl.collect_base_link_cloud(5)
    if xy is None or len(xy) < 30:
        print(f"S{seed}A{{att+1}} SKIP cloud", flush=True); continue
    lines = cl.detect_lines(xy, xy3)
    if len(lines) < 2:
        print(f"S{seed}A{{att+1}} SKIP lines", flush=True); continue
    pairs, _ = cl.find_orthogonal_pairs(lines)
    if not pairs:
        print(f"S{seed}A{{att+1}} SKIP pairs", flush=True); continue
    allr = []
    for p in pairs:
        allr.extend(cl.generate_candidates_for_pair(p["li"], p["lj"], xy3, ang, ran))
    if not allr:
        print(f"S{seed}A{{att+1}} SKIP cands", flush=True); continue
    allr.sort(key=lambda c: -c["score"])
    cands = [fv.Candidate(i, c["x"], c["y"], c["yaw"], "corner", c["score"]) for i, c in enumerate(allr[:20])]

    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    import tf2_ros, tf.transformations as tft

    tf_buf = tf2_ros.Buffer(); tf_lis = tf2_ros.TransformListener(tf_buf)
    rospy.sleep(0.5)
    tf_trans, tf_rot = None, None
    try:
        t = tf_buf.lookup_transform("base_link", "radar", rospy.Time(0), rospy.Duration(2.0))
        tf_trans = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        q = t.transform.rotation
        tf_rot = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    except: pass
    frames = []
    for _ in range(5):
        try:
            msg = rospy.wait_for_message("/lidar/points", PointCloud2, timeout=2.0)
            pts = np.array(list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)))
            r = np.sqrt(pts[:,0]**2+pts[:,1]**2)
            pts = pts[(r>0.35)&(r<20)]
            if len(pts)>0: frames.append(pts)
            rospy.sleep(0.1)
        except: continue
    if len(frames)<3:
        print(f"S{seed}A{{att+1}} SKIP frames={{len(frames)}}", flush=True); continue

    vr = fv.validate_candidates_at_fix(cands, frames, gt)
    sel = [r for r in vr if r.selected]
    for r in vr:
        results.append({{
            "seed": {seed}, "attempt": att+1, "candidate_id": r.candidate.id,
            "source": r.candidate.source, "x": r.candidate.x, "y": r.candidate.y,
            "yaw_deg": math.degrees(r.candidate.yaw), "orig_score": r.candidate.orig_score,
            "right_count": r.right.count, "right_med": round(r.right.med_residual,4),
            "right_p90": round(r.right.p90_residual,4), "right_span": round(r.right.span,3),
            "right_density": r.right.density_bins, "right_ok": r.right.ok,
            "back_count": r.back.count, "back_med": round(r.back.med_residual,4),
            "back_p90": round(r.back.p90_residual,4), "back_span": round(r.back.span,3),
            "back_density": r.back.density_bins, "back_ok": r.back.ok,
            "fix_score": round(r.fix_score,1) if r.fix_score else 0,
            "selected": r.selected, "reject_reason": r.reject_reason or "",
            "gt_x": round(gt[0],3) if gt else None, "gt_y": round(gt[1],3) if gt else None,
            "gt_yaw_deg": round(math.degrees(gt[2]),1) if gt else None,
            "pos_err": round(r.pos_err,3) if r.pos_err else None,
            "yaw_err_deg": round(math.degrees(r.yaw_err),1) if r.yaw_err else None,
            "pass": (r.pos_err is not None and r.pos_err<0.25 and r.yaw_err is not None and r.yaw_err<0.175) if r.selected else False,
        }})
    sid = sel[0].candidate.id if sel else None
    print(f"S{seed}A{{att+1}} {{'SELECTED#'+str(sid) if sel else 'NO_SEL'}} n_ok={{sum(1 for r in vr if r.right.ok and r.back.ok)}}", flush=True)

os.makedirs("/tmp/nav_test/logs", exist_ok=True)
with open(f"/tmp/nav_test/logs/swp_t{str(threshold).replace('.','p')}_s{seed}.json", "w") as f:
    json.dump(results, f)
print(f"__DONE__ {{len(results)}}", flush=True)
launcher.stop()
'''

    script_path = f"/tmp/swp_seed{seed}.py"
    with open(script_path, "w") as f:
        f.write(script)
    subprocess.run(f"docker cp {script_path} {C}:/tmp/", shell=True, capture_output=True)

    out = subprocess.run(
        f"docker exec -e DISPLAY=:0 {C} bash -c 'source /opt/ros/noetic/setup.bash && source /root/kuavo_ws/devel/setup.bash && export LD_LIBRARY_PATH=/opt/drake/lib:/root/kuavo_ws/installed/lib:$LD_LIBRARY_PATH && export ROBOT_VERSION=52 DISPLAY=:0 PYTHONUNBUFFERED=1 && cd /tmp/nav_test && timeout 360 python3 -u {script_path} 2>/dev/null'",
        shell=True, capture_output=True, text=True, timeout=480
    )

    # Read results from container file
    json_name = f"swp_t{str(threshold).replace('.','p')}_s{seed}.json"
    cp = subprocess.run(f"docker cp {C}:/tmp/nav_test/logs/{json_name} /tmp/ 2>/dev/null", shell=True, capture_output=True)
    try:
        with open(f"/tmp/{json_name}") as f:
            return json.load(f)
    except:
        return []


# ═══ Main ═══
print("=" * 70)
print("PARAMETER SWEEP: MAX_MED_RESIDUAL ∈ {0.05, 0.07, 0.10}")
print(f"Seeds: {SEEDS} × {ATTEMPTS} attempts × {len(THRESHOLDS)} thresholds")
print("=" * 70)

all_rows = []
summary = {t: {} for t in THRESHOLDS}

for threshold in THRESHOLDS:
    print(f"\n{'─'*60}\nTHRESHOLD = {threshold:.2f}\n{'─'*60}")

    for seed in SEEDS:
        print(f"  seed {seed}...", end=" ", flush=True)
        rows = run_seed(threshold, seed)
        if rows:
            for r in rows:
                r["threshold"] = threshold
            all_rows.extend(rows)

            # Per-attempt summary
            att_results = []
            for a in range(1, ATTEMPTS + 1):
                ar = [r for r in rows if r["attempt"] == a]
                sel = [r for r in ar if r["selected"]]
                if not ar: att_results.append("SKIP")
                elif any(r["pass"] for r in sel): att_results.append("PASS")
                elif sel: att_results.append("SEL")
                else: att_results.append("NO_SEL")
            summary[threshold][seed] = {"attempts": att_results, "pass": att_results.count("PASS")}
            print(f"OK ({len(rows)} rows)  {att_results}")
        else:
            summary[threshold][seed] = {"attempts": ["CRASH"]*ATTEMPTS, "pass": 0}
            print("CRASH")

        subprocess.run(f"docker exec {C} pkill -9 -f ros 2>/dev/null", shell=True, capture_output=True)
        time.sleep(5)

# ═══ Full CSV ═══
csv_path = os.path.join(REPORT_DIR, "param_sweep_full.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader(); w.writerows(all_rows)
jsonl_path = os.path.join(REPORT_DIR, "param_sweep_full.jsonl")
with open(jsonl_path, "w") as f:
    for row in all_rows: f.write(json.dumps(row) + "\n")

# ═══ Summary Table ═══
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for threshold in THRESHOLDS:
    th_data = summary[threshold]
    total_p = sum(th_data[s]["pass"] for s in SEEDS)
    total_t = sum(len([a for a in th_data[s]["attempts"] if a != "SKIP" and a != "CRASH"]) for s in SEEDS)
    print(f"\nMAX_MED_RESIDUAL = {threshold:.2f}  →  {total_p}/{total_t} PASS  "
          f"({'✅' if total_p >= 13 else '❌'})")
    print(f"{'seed':<6} {'att1':<8} {'att2':<8} {'att3':<8} {'pass':<6}")
    for s in SEEDS:
        sd = th_data.get(s, {"attempts": ["???"]*ATTEMPTS, "pass": 0})
        a = sd["attempts"]
        print(f"{s:<6} {a[0]:<8} {a[1] if len(a)>1 else 'N/A':<8} "
              f"{a[2] if len(a)>2 else 'N/A':<8} {sd['pass']}/{ATTEMPTS}")

    # NO_SEL count
    no_sel = sum(1 for s in SEEDS for a in th_data[s]["attempts"] if a == "NO_SEL")
    swap_count = 0  # selected but wrong (pos_err > 0.25)
    for s in SEEDS:
        srows = [r for r in all_rows if r["threshold"] == threshold and r["seed"] == s]
        swap_count += sum(1 for r in srows if r["selected"] and not r["pass"])
    print(f"NO_SEL: {no_sel}  SELECTED_WRONG: {swap_count}")

print(f"\nFull data: {csv_path} ({len(all_rows)} rows)")
print(jsonl_path)
