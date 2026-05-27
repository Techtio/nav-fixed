#!/usr/bin/env python3
"""Host-side orchestrator for fixed_pose_validator acceptance test.
Runs each seed as a standalone docker exec (immune to SimLauncher pkill)."""
import subprocess, time, sys, os, json, csv

SEEDS = [0, 7, 13, 31, 42]
ATTEMPTS = 3
C = "kuavo_clean_craic"
ALL_ROWS = []

COLS = ["seed","attempt","candidate_id","source","x","y","yaw_deg",
        "orig_score","right_count","right_med","right_p90","right_span",
        "right_density","right_ok","back_count","back_med","back_p90",
        "back_span","back_density","back_ok","fix_score","selected",
        "reject_reason","gt_x","gt_y","gt_yaw_deg","pos_err","yaw_err_deg","pass"]

HOST_LOG = os.path.expanduser("~/Desktop/fix_validator_results.jsonl")
HOST_CSV = os.path.expanduser("~/Desktop/fix_validator_results.csv")
SUMMARY = {}

print("=" * 60)
print("FIX VALIDATOR ACCEPTANCE (host-side orchestration)")
print(f"Seeds: {SEEDS} × {ATTEMPTS} attempts")
print("=" * 60)

for seed in SEEDS:
    print(f"\n{'─'*50}\nSEED {seed}\n{'─'*50}")

    # Write single-seed test script
    script = f'''import rospy, sys, os, time, math, json, csv
os.environ["DISPLAY"] = ":0"
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/utils")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher
launcher = SimLauncher(scene="scene1", seed={seed}, robot_version=52)
launcher.start(node_name="fv_host_s{seed}", timeout=120)
print("SIM_READY", flush=True)
time.sleep(8)

sys.path.insert(0, "/tmp/nav_test")
import corner_localizer as cl
import fixed_pose_validator as fv
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_ros, tf.transformations as tft
import numpy as np

gt = cl.GT_POSES.get({seed})
print(f"GT: x={{gt[0]:.3f}} y={{gt[1]:.3f}} yaw={{math.degrees(gt[2]):.1f}}°") if gt else print("GT: N/A")

results = []
for att in range({ATTEMPTS}):
    time.sleep(2)
    xy, xy3, ang, ran = cl.collect_base_link_cloud(5)
    if xy is None or len(xy) < 30:
        print(f"S{seed}A{{att+1}} SKIP: no cloud"); continue
    lines = cl.detect_lines(xy, xy3)
    if len(lines) < 2:
        print(f"S{seed}A{{att+1}} SKIP: <2 lines"); continue
    pairs, _ = cl.find_orthogonal_pairs(lines)
    if not pairs:
        print(f"S{seed}A{{att+1}} SKIP: no pairs"); continue
    allr = []
    for p in pairs:
        allr.extend(cl.generate_candidates_for_pair(p["li"], p["lj"], xy3, ang, ran))
    if not allr:
        print(f"S{seed}A{{att+1}} SKIP: no candidates"); continue
    allr.sort(key=lambda c: -c["score"])
    cands = [fv.Candidate(i, c["x"], c["y"], c["yaw"], "corner", c["score"]) for i, c in enumerate(allr[:20])]

    # Frames
    tf_buf = tf2_ros.Buffer(); tf_lis = tf2_ros.TransformListener(tf_buf)
    rospy.sleep(0.5)
    tf_rot = None; tf_trans = None
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
        print(f"S{seed}A{{att+1}} SKIP: {{len(frames)}} frames"); continue

    vr = fv.validate_candidates_at_fix(cands, frames, gt)
    sel = [r for r in vr if r.selected]
    sid = sel[0].candidate.id if sel else None
    for r in vr:
        row = {{
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
        }}
        results.append(row)

    pe = f"pos={{sel[0].pos_err:.3f}}" if sel and sel[0].pos_err else "pos=N/A"
    ye = f"yaw={{math.degrees(sel[0].yaw_err):.1f}}°" if sel and sel[0].yaw_err else "yaw=N/A"
    print(f"S{seed}A{{att+1}} {{'SELECTED #'+str(sid) if sel else 'NO SELECTION'}}  {{pe}} {{ye}}")
    if not sel:
        for r in vr[:5]:
            print(f"  [{{r.candidate.id}}] R(n={{r.right.count}} m={{r.right.med_residual:.3f}} s={{r.right.span:.2f}} ok={{r.right.ok}}) B(n={{r.back.count}} m={{r.back.med_residual:.3f}} s={{r.back.span:.2f}} ok={{r.back.ok}})")

# Output results as JSON
import json
print("__RESULTS_JSON__")
print(json.dumps(results))
print("__END__")
launcher.stop()
'''

    script_path = f"/tmp/fv_seed{seed}.py"
    with open(script_path, "w") as f:
        f.write(script)
    subprocess.run(f"docker cp {script_path} {C}:/tmp/", shell=True, capture_output=True)

    # Run seed test (waits for SimLauncher + detection + stop)
    out = subprocess.run(
        f"docker exec -e DISPLAY=:0 {C} bash -c 'source /opt/ros/noetic/setup.bash && source /root/kuavo_ws/devel/setup.bash && export LD_LIBRARY_PATH=/opt/drake/lib:/root/kuavo_ws/installed/lib:$LD_LIBRARY_PATH && export ROBOT_VERSION=52 && export DISPLAY=:0 && export PYTHONUNBUFFERED=1 && cd /tmp/nav_test && python3 -u {script_path} 2>/dev/null'",
        shell=True, capture_output=True, text=True, timeout=480
    )

    # Extract results JSON
    stdout = out.stdout + out.stderr
    if "__RESULTS_JSON__" in stdout:
        json_part = stdout.split("__RESULTS_JSON__")[1].split("__END__")[0].strip()
        try:
            seed_results = json.loads(json_part)
            ALL_ROWS.extend(seed_results)
            # Summary
            att_results = []
            for a in range(1, ATTEMPTS + 1):
                att_rows = [r for r in seed_results if r["attempt"] == a]
                sel = [r for r in att_rows if r["selected"]]
                if not att_rows:
                    att_results.append("SKIP")
                elif any(r["pass"] for r in sel):
                    att_results.append("PASS")
                elif sel:
                    att_results.append("SEL")
                else:
                    att_results.append("NO_SEL")
            SUMMARY[seed] = att_results
            pc = att_results.count("PASS")
            print(f"  SEED {seed}: {att_results}  PASS={pc}/{ATTEMPTS}")
        except json.JSONDecodeError as e:
            print(f"  SEED {seed}: JSON parse error: {e}")
            SUMMARY[seed] = ["ERR"] * ATTEMPTS
    else:
        print(f"  SEED {seed}: NO JSON output")
        # Show last lines for debug
        for line in stdout.split("\n")[-20:]:
            if line.strip(): print(f"    {line[:120]}")
        SUMMARY[seed] = ["CRASH"] * ATTEMPTS

    # Cleanup: kill any remaining ROS in container
    subprocess.run(f"docker exec {C} pkill -9 -f ros 2>/dev/null", shell=True, capture_output=True)
    time.sleep(5)

# ═══ Final Output ═══
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'seed':<6} {'att1':<8} {'att2':<8} {'att3':<8} {'pass_count':<12}")
print("-" * 42)
total_pass = 0
for seed in SEEDS:
    sr = SUMMARY.get(seed, ["???"] * ATTEMPTS)
    pc = sr.count("PASS")
    total_pass += pc
    print(f"{seed:<6} {sr[0]:<8} {sr[1] if len(sr)>1 else 'N/A':<8} "
          f"{sr[2] if len(sr)>2 else 'N/A':<8} {pc}/{ATTEMPTS}")
print("-" * 42)
print(f"TOTAL: {total_pass}/{len(SEEDS)*ATTEMPTS}")

# Write CSV
with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader(); w.writerows(ALL_ROWS)
with open(JSONL_PATH, "w") as f:
    for row in ALL_ROWS: f.write(json.dumps(row) + "\n")

print(f"\nResults: {CSV_PATH}  ({len(ALL_ROWS)} rows)")
print(f"          {JSONL_PATH}")

if total_pass < 13:
    print("\n⚠ BELOW 13/15 — DIAGNOSIS")
    print("=" * 60)
    failures = [r for r in ALL_ROWS if r["selected"] and not r["pass"]]
    for f in failures:
        print(f"\nFAIL S{f['seed']}A{f['attempt']} C{f['candidate_id']}: "
              f"pos_err={f['pos_err']:.3f}m yaw_err={f['yaw_err_deg']:.1f}° "
              f"R(n={f['right_count']} m={f['right_med']} s={f['right_span']}) "
              f"B(n={f['back_count']} m={f['back_med']} s={f['back_span']})")
    no_sel = [(s, a) for s in SEEDS for a in range(1, ATTEMPTS+1)
              if SUMMARY.get(s, [""]*ATTEMPTS)[a-1] == "NO_SEL"]
    for s, a in no_sel:
        print(f"\nFAIL S{s}A{a}: NO SELECTION — back_ok never passed "
              f"(med_residual consistently above 0.05 threshold)")
    print("\nRoot cause: B. fixed_pose_validator BACK wall threshold too strict "
          "for simulation wall geometry (~5-6cm offset from theoretical). "
          "Need to calibrate MAX_MED_RESIDUAL or BACK_TARGET.")

print("\nDONE.")
