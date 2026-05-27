#!/usr/bin/env python3
"""
run_fixed_validator_5x3.py — Batch runner for fixed_pose_validator acceptance test.
DOES NOT modify fixed_pose_validator.py scoring logic.
Orchestrates: SimLauncher per seed → corner_localizer → validator → log.
"""
import subprocess, time, sys, os, json, csv, math
from pathlib import Path

SEEDS = [0, 7, 13, 31, 42]
ATTEMPTS = 3
N_FRAMES = 5
TOPK = 20
LOG_DIR = Path("/tmp/nav_test/logs/fix_validator_5x3")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ═══ Import validator (no modification) ═══
sys.path.insert(0, "/tmp/nav_test")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/utils")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/lib")
import fixed_pose_validator as fv
import corner_localizer as cl
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_ros, tf.transformations as tft
import numpy as np

CSV_PATH = LOG_DIR / "results.csv"
JSONL_PATH = LOG_DIR / "results.jsonl"
COLS = [
    "seed", "attempt", "candidate_id", "source", "x", "y", "yaw_deg",
    "orig_score", "right_count", "right_med", "right_p90", "right_span",
    "right_density", "right_ok", "back_count", "back_med", "back_p90",
    "back_span", "back_density", "back_ok", "fix_score", "selected",
    "reject_reason", "gt_x", "gt_y", "gt_yaw_deg", "pos_err", "yaw_err_deg", "pass"
]

all_rows = []

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "run.log", "a") as f:
        f.write(line + "\n")

def collect_candidates_and_frames():
    """Call corner_localizer pipeline → return (candidates, lidar_frames)."""
    xy_vox, xy3_filt, angles, ranges = cl.collect_base_link_cloud(N_FRAMES)
    if xy_vox is None or len(xy_vox) < 30:
        return [], []

    lines = cl.detect_lines(xy_vox, xy3_filt)
    if len(lines) < 2:
        return [], []
    pairs, _ = cl.find_orthogonal_pairs(lines)
    if not pairs:
        return [], []

    all_raw = []
    for p in pairs:
        all_raw.extend(cl.generate_candidates_for_pair(
            p["li"], p["lj"], xy3_filt, angles, ranges))
    if not all_raw:
        return [], []

    all_raw.sort(key=lambda c: -c["score"])
    top = all_raw[:TOPK]

    candidates = []
    for i, c in enumerate(top):
        candidates.append(fv.Candidate(
            id=i, x=c["x"], y=c["y"], yaw=c["yaw"],
            source="corner", orig_score=c["score"]))

    # Collect lidar frames for FIX validation
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)

    tf_trans, tf_rot = None, None
    try:
        t = tf_buffer.lookup_transform("base_link", "radar",
                                       rospy.Time(0), rospy.Duration(2.0))
        tf_trans = np.array([t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z])
        q = t.transform.rotation
        tf_rot = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    except Exception:
        pass

    frames = []
    for _ in range(N_FRAMES):
        try:
            msg = rospy.wait_for_message("/lidar/points", PointCloud2, timeout=2.0)
            pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                               skip_nans=True)))
            if len(pts) == 0:
                continue
            fid = msg.header.frame_id
            if fid and fid != "base_link" and tf_rot is not None:
                pts[:, :3] = pts[:, :3] @ tf_rot.T + tf_trans
            r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            pts = pts[(r > 0.35) & (r < 20.0)]
            if len(pts) > 0:
                frames.append(pts)
            rospy.sleep(0.1)
        except Exception:
            continue
    return candidates, frames


def run_one_test(seed, attempt, gt):
    """Single attempt: collect candidates, validate, record."""
    cands, frames = collect_candidates_and_frames()
    if not cands or len(frames) < 3:
        log(f"  S{seed}A{attempt} SKIP: cands={len(cands)} frames={len(frames)}")
        return None

    results = fv.validate_candidates_at_fix(cands, frames, gt)
    sel = [r for r in results if r.selected]
    selected_id = sel[0].candidate.id if sel else None

    rows = []
    for r in results:
        row = {
            "seed": seed, "attempt": attempt,
            "candidate_id": r.candidate.id, "source": r.candidate.source,
            "x": r.candidate.x, "y": r.candidate.y,
            "yaw_deg": math.degrees(r.candidate.yaw),
            "orig_score": r.candidate.orig_score,
            "right_count": r.right.count,
            "right_med": round(r.right.med_residual, 4),
            "right_p90": round(r.right.p90_residual, 4),
            "right_span": round(r.right.span, 3),
            "right_density": r.right.density_bins,
            "right_ok": r.right.ok,
            "back_count": r.back.count,
            "back_med": round(r.back.med_residual, 4),
            "back_p90": round(r.back.p90_residual, 4),
            "back_span": round(r.back.span, 3),
            "back_density": r.back.density_bins,
            "back_ok": r.back.ok,
            "fix_score": round(r.fix_score, 1) if r.fix_score else 0,
            "selected": r.selected,
            "reject_reason": r.reject_reason or "",
            "gt_x": round(gt[0], 3) if gt else None,
            "gt_y": round(gt[1], 3) if gt else None,
            "gt_yaw_deg": round(math.degrees(gt[2]), 1) if gt else None,
            "pos_err": round(r.pos_err, 3) if r.pos_err else None,
            "yaw_err_deg": round(math.degrees(r.yaw_err), 1) if r.yaw_err else None,
            "pass": (r.pos_err is not None and r.pos_err < 0.25 and
                     r.yaw_err is not None and r.yaw_err < 0.175) if r.selected else False,
        }
        rows.append(row)

    # Summary line
    if sel:
        best = sel[0]
        pe = f"pos={best.pos_err:.3f}" if best.pos_err else "pos=N/A"
        ye = f"yaw={math.degrees(best.yaw_err):.1f}°" if best.yaw_err else "yaw=N/A"
        log(f"  S{seed}A{attempt} SELECTED #{best.candidate.id}  {pe} {ye}  "
            f"R(n={best.right.count} m={best.right.med_residual:.3f} "
            f"s={best.right.span:.2f} ok={best.right.ok})  "
            f"B(n={best.back.count} m={best.back.med_residual:.3f} "
            f"s={best.back.span:.2f} ok={best.back.ok})")
    else:
        log(f"  S{seed}A{attempt} NO SELECTION")
        for r in results[:5]:
            log(f"    [{r.candidate.id}] right_ok={r.right.ok} back_ok={r.back.ok} "
                f"R(n={r.right.count} m={r.right.med_residual:.3f}) "
                f"B(n={r.back.count} m={r.back.med_residual:.3f})")

    return rows, selected_id


def start_simulation(seed):
    """Start SimLauncher for given seed. Returns once simulation is ready."""
    os.environ["DISPLAY"] = ":0"
    
    # Let SimLauncher handle its own cleanup (it does pkill internally)
    from sim_launcher import SimLauncher
    launcher = SimLauncher(scene="scene1", seed=seed, robot_version=52)
    launcher.start(node_name=f"fv_batch_s{seed}", timeout=120)
    log(f"SimLauncher started seed={seed}")
    time.sleep(8)  # let TF, EKF, controllers stabilize

    # Init ROS node for this seed (after SimLauncher's roslaunch is running)
    try:
        rospy.init_node(f"fv_test_s{seed}", anonymous=True)
    except rospy.exceptions.ROSException:
        pass
    time.sleep(2)

    # Kill interference nodes
    for node in ["/humanoid_quest_control_with_arm", "/humanoid_keyboard_control",
                 "/humanoid_teleop_control", "/humanoid_VR_hand_control"]:
        try:
            subprocess.run(["rosnode", "kill", node], capture_output=True, timeout=5)
        except Exception:
            pass
    time.sleep(2)

    return launcher


def stop_simulation(launcher=None):
    """Gracefully stop simulation — NEVER use pkill (would kill self)."""
    if launcher:
        try:
            launcher.stop()
        except Exception:
            pass
    time.sleep(3)
    # Only kill specific rosmaster if launcher.stop() missed it
    try:
        subprocess.run(["rosnode", "kill", "-a"], capture_output=True, timeout=10)
    except Exception:
        pass
    time.sleep(3)


# ═══ Main ═══
log("=" * 60)
log("FIXED POSE VALIDATOR ACCEPTANCE TEST (5×3)")
log(f"Seeds: {SEEDS}  Attempts: {ATTEMPTS}  Frames: {N_FRAMES}  TopK: {TOPK}")
log(f"Log dir: {LOG_DIR}")
log("=" * 60)

summary = {}  # seed -> [attempt_results]

for seed in SEEDS:
    log(f"\n{'─' * 50}")
    log(f"SEED {seed}")
    log(f"{'─' * 50}")

    launcher = start_simulation(seed)
    gt = cl.GT_POSES.get(seed)
    if gt:
        log(f"GT: x={gt[0]:.3f} y={gt[1]:.3f} yaw={math.degrees(gt[2]):.1f}°")

    seed_results = []
    for att in range(ATTEMPTS):
        rows, sel_id = run_one_test(seed, att + 1, gt)
        if rows:
            all_rows.extend(rows)
            has_sel = any(r["selected"] for r in rows)
            has_pass = any(r["pass"] for r in rows)
            seed_results.append("PASS" if has_pass else ("SEL" if has_sel else "NO_SEL"))
        else:
            seed_results.append("SKIP")

    summary[seed] = seed_results
    # Incremental CSV write after each seed
    if all_rows:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        with open(JSONL_PATH, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
    stop_simulation(launcher)
    log(f"SEED {seed} DONE ({seed_results})")

# ═══ Write CSV/JSONL ═══
with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_rows)
log(f"\nCSV written: {CSV_PATH}  ({len(all_rows)} rows)")

with open(JSONL_PATH, "w") as f:
    for row in all_rows:
        f.write(json.dumps(row) + "\n")
log(f"JSONL written: {JSONL_PATH}")

# ═══ Summary ═══
log("\n" + "=" * 60)
log("RESULTS SUMMARY")
log("=" * 60)
log(f"{'seed':<6} {'att1':<8} {'att2':<8} {'att3':<8} {'pass_count':<12}")
log("-" * 42)

total_pass = 0
for seed in SEEDS:
    sr = summary.get(seed, ["SKIP"] * ATTEMPTS)
    pc = sr.count("PASS")
    total_pass += pc
    log(f"{seed:<6} {sr[0]:<8} {sr[1] if len(sr)>1 else 'N/A':<8} "
        f"{sr[2] if len(sr)>2 else 'N/A':<8} {pc}/{ATTEMPTS}")

log("-" * 42)
log(f"TOTAL: {total_pass}/{len(SEEDS)*ATTEMPTS}")
log("=" * 60)

if total_pass < 13:
    log("\n⚠ BELOW 13/15 — FAILURE DIAGNOSIS")
    log("=" * 60)
    # Collect failures
    failures = [r for r in all_rows if r["selected"] and not r["pass"]]
    no_sel = [(s, a) for s in SEEDS for a in range(1, ATTEMPTS+1)
              if not any(r["seed"] == s and r["attempt"] == a and r["selected"]
                        for r in all_rows)]
    for f in failures:
        log(f"\nFAIL S{f['seed']}A{f['attempt']} C{f['candidate_id']}: "
            f"pos={f['pos_err']:.3f}m yaw={f['yaw_err_deg']:.1f}° "
            f"R(n={f['right_count']} m={f['right_med']} s={f['right_span']}) "
            f"B(n={f['back_count']} m={f['back_med']} s={f['back_span']})")
        # Find GT-closest candidate
        best_gt_row = None
        best_gt_dist = 999
        for r in all_rows:
            if r["seed"] == f["seed"] and r["attempt"] == f["attempt"]:
                d = ((r["x"] - r["gt_x"])**2 + (r["y"] - r["gt_y"])**2)**0.5
                if d < best_gt_dist:
                    best_gt_dist = d
                    best_gt_row = r
        if best_gt_row and best_gt_row["candidate_id"] != f["candidate_id"]:
            log(f"  GT-closest: C{best_gt_row['candidate_id']} "
                f"err={best_gt_dist:.3f}m "
                f"R(n={best_gt_row['right_count']} m={best_gt_row['right_med']} "
                f"s={best_gt_row['right_span']} ok={best_gt_row['right_ok']}) "
                f"B(n={best_gt_row['back_count']} m={best_gt_row['back_med']} "
                f"s={best_gt_row['back_span']} ok={best_gt_row['back_ok']}) "
                f"selected={best_gt_row['selected']}")
            log(f"  Root cause: {'B. validator 误选' if best_gt_row['right_ok'] and best_gt_row['back_ok'] else 'C. 点云/TF 异常' if best_gt_dist < 0.1 else 'A. corner_localizer 未生成正确候选'}")
    for s, a in no_sel:
        log(f"\nFAIL S{s}A{a}: NO SELECTION")

log("\nDONE. Exit.")
