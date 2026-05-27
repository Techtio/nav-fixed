#!/usr/bin/env python3
"""
fixed_pose_validator.py — Dominant-peak FIX-point candidate validator.

v3: Geometric-error-first scoring. Count capped. AMBIGUOUS rejection.
Peak position determines wall_ok, not raw residual.  
NO_SEL > wrong selection.

Usage:
  python3 fixed_pose_validator.py --seeds 0,7,13,31,42 --attempts 3
"""

import sys, os, time, math, json, csv, argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# ═══ Constants ═══
FIX_X   = 5.01;  FIX_Y   = 3.50;  FIX_YAW = math.pi
NORTH_Y = 4.50;  EAST_X  = 6.25
RIGHT_TARGET = -1.00   # FIX view: y of north wall
BACK_TARGET  = -1.24   # FIX view: x of east wall

PEAK_SEARCH_HALF = 0.45
PEAK_BIN = 0.02
STRIP_HALF = 0.12

RIGHT_PEAK_ERR_MAX = 0.10
BACK_PEAK_ERR_MAX  = 0.10
MIN_RIGHT_COUNT = 800
MIN_BACK_COUNT  = 500
MIN_WALL_SPAN   = 0.60

MIN_SCORE_MARGIN = 8.0   # reject if top1-top2 < this

N_FRAMES    = 5
VOXEL_SIZE  = 0.03
R_MIN       = 0.35
R_MAX       = 20.0

# ═══ Data ═══

@dataclass
class Candidate:
    id: int;  x: float;  y: float;  yaw: float
    source: str = "";  orig_score: float = 0.0
    @property
    def yaw_deg(self): return math.degrees(self.yaw) % 360

@dataclass
class WallMetrics:
    ok: bool = False
    peak: Optional[float] = None
    peak_err: float = 999.0
    count: int = 0
    med: float = 999.0
    p90: float = 999.0
    span: float = 0.0
    density: float = 0.0

@dataclass
class ValidResult:
    candidate: Candidate
    right: WallMetrics = field(default_factory=WallMetrics)
    back:  WallMetrics = field(default_factory=WallMetrics)
    fix_score: float = 0.0
    selected: bool = False
    reject_reason: str = ""
    gt_x: Optional[float] = None
    gt_y: Optional[float] = None
    gt_yaw: Optional[float] = None
    gt_ok: bool = False

    @property
    def pos_err(self):
        if self.gt_x is None: return None
        return math.hypot(self.candidate.x - self.gt_x, self.candidate.y - self.gt_y)
    @property
    def yaw_err(self):
        if self.gt_yaw is None: return None
        d = (self.candidate.yaw - self.gt_yaw) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    def row_dict(self, seed, attempt):
        r = {
            "seed": seed, "attempt": attempt,
            "candidate_id": self.candidate.id, "source": self.candidate.source,
            "x": round(self.candidate.x, 4), "y": round(self.candidate.y, 4),
            "yaw_deg": round(self.candidate.yaw_deg, 2),
            "orig_score": round(self.candidate.orig_score, 4),
            "right_peak": round(self.right.peak, 4) if self.right.peak else None,
            "right_peak_err": round(self.right.peak_err, 4),
            "right_count": self.right.count,
            "right_ok": self.right.ok,
            "back_peak": round(self.back.peak, 4) if self.back.peak else None,
            "back_peak_err": round(self.back.peak_err, 4),
            "back_count": self.back.count,
            "back_ok": self.back.ok,
            "fix_score": round(self.fix_score, 2), "selected": self.selected,
            "reject_reason": self.reject_reason,
            "gt_ok": self.gt_ok,
        }
        if self.gt_x is not None:
            r.update({"gt_x": round(self.gt_x, 4), "gt_y": round(self.gt_y, 4),
                      "gt_yaw_deg": round(math.degrees(self.gt_yaw), 2),
                      "pos_err": round(self.pos_err, 4) if self.pos_err else None,
                      "yaw_err_deg": round(math.degrees(self.yaw_err), 2) if self.yaw_err else None})
        return r


# ═══ Dominant Peak ═══

def dominant_wall_peak(pts_fix, axis, target, along_axis):
    empty = lambda: {"ok": False, "peak": None, "peak_err": 999.0,
                     "count": 0, "med": 999.0, "p90": 999.0, "span": 0.0, "density": 0.0}
    if pts_fix is None or len(pts_fix) < 50: return empty()
    coord = pts_fix[:, axis]
    mask = np.abs(coord - target) <= PEAK_SEARCH_HALF
    pts = pts_fix[mask]
    if len(pts) < 50:
        return {"ok": False, "peak": None, "peak_err": 999.0, "count": len(pts),
                "med": 999.0, "p90": 999.0, "span": 0.0, "density": 0.0}
    vals = pts[:, axis]
    bins = np.arange(target - PEAK_SEARCH_HALF, target + PEAK_SEARCH_HALF + PEAK_BIN, PEAK_BIN)
    hist, edges = np.histogram(vals, bins=bins)
    kernel = np.ones(5, dtype=float) / 5.0
    smooth = np.convolve(hist, kernel, mode="same")
    idx = int(np.argmax(smooth))
    peak = 0.5 * (edges[idx] + edges[idx + 1])
    peak_err = abs(peak - target)
    strip = np.abs(pts[:, axis] - peak) <= STRIP_HALF
    wall_pts = pts[strip]
    if len(wall_pts) == 0:
        return {"ok": False, "peak": float(peak), "peak_err": float(peak_err),
                "count": 0, "med": 999.0, "p90": 999.0, "span": 0.0, "density": 0.0}
    residual = np.abs(wall_pts[:, axis] - target)
    count = int(len(wall_pts))
    med = float(np.median(residual))
    p90 = float(np.percentile(residual, 90))
    along = wall_pts[:, along_axis]
    span = float(np.percentile(along, 90) - np.percentile(along, 10))
    density = count / max(span, 0.1)
    return {"ok": False, "peak": float(peak), "peak_err": float(peak_err),
            "count": count, "med": med, "p90": p90, "span": span, "density": float(density)}


def score_fix_view(pts_fix):
    """Geometric-error-first scoring. Count capped, peak_err dominates."""
    right = dominant_wall_peak(pts_fix, axis=1, target=RIGHT_TARGET, along_axis=0)
    back  = dominant_wall_peak(pts_fix, axis=0, target=BACK_TARGET,  along_axis=1)

    right_ok = (right["peak_err"] <= RIGHT_PEAK_ERR_MAX
                and right["count"] >= MIN_RIGHT_COUNT
                and right["span"] >= MIN_WALL_SPAN)
    back_ok  = (back["peak_err"] <= BACK_PEAK_ERR_MAX
                and back["count"] >= MIN_BACK_COUNT
                and back["span"] >= MIN_WALL_SPAN)

    right["ok"] = right_ok
    back["ok"]  = back_ok

    if not (right_ok and back_ok):
        return {"ok": False, "score": -1e9, "right": right, "back": back}

    score = 0.0
    score += 80.0 if right_ok else -200.0
    score += 80.0 if back_ok  else -200.0
    # Peak error penalty — quadratic, dominates
    score -= 120.0 * (right["peak_err"] / 0.10) ** 2
    score -= 120.0 * (back["peak_err"]  / 0.10) ** 2
    # Count bonus — capped, log-scale to prevent domination
    score += 15.0 * min(right["count"], 1800) / 1800.0
    score += 15.0 * min(back["count"],  1600) / 1600.0
    # Span bonus — capped
    score += 10.0 * min(right["span"], 2.0) / 2.0
    score += 10.0 * min(back["span"],  2.0) / 2.0
    return {"ok": True, "score": float(score), "right": right, "back": back}


# ═══ Transform ═══

def transform_to_world(pts, x, y, yaw):
    if len(pts) == 0: return pts
    c, s = math.cos(yaw), math.sin(yaw)
    out = pts.copy()
    out[:, 0] = pts[:, 0] * c - pts[:, 1] * s + x
    out[:, 1] = pts[:, 0] * s + pts[:, 1] * c + y
    return out

def world_to_fix(pts_w):
    if len(pts_w) == 0: return pts_w
    out = pts_w.copy()
    out[:, 0] -= FIX_X; out[:, 1] -= FIX_Y
    c, s = math.cos(-FIX_YAW), math.sin(-FIX_YAW)
    out[:, 0] = out[:, 0] * c - out[:, 1] * s
    out[:, 1] = out[:, 0] * s + out[:, 1] * c
    return out

def voxel_downsample(pts, size=VOXEL_SIZE):
    if len(pts) < 2: return pts
    vox = np.floor(pts[:, :3] / size).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return pts[idx]


# ═══ Validate ═══

def validate_candidates_at_fix(
    candidates: List[Candidate],
    points_frames: List[np.ndarray],
    gt: Optional[Tuple[float, float, float]] = None
) -> List[ValidResult]:
    if not points_frames: return []
    merged = np.vstack(points_frames)
    merged = voxel_downsample(merged)

    results = []
    for c in candidates:
        p_world = transform_to_world(merged, c.x, c.y, c.yaw)
        p_fix = world_to_fix(p_world)
        r = score_fix_view(p_fix)

        right = WallMetrics(ok=r["right"]["ok"], peak=r["right"]["peak"],
                            peak_err=r["right"]["peak_err"],
                            count=r["right"]["count"], med=r["right"]["med"],
                            p90=r["right"]["p90"], span=r["right"]["span"],
                            density=r["right"]["density"])
        back = WallMetrics(ok=r["back"]["ok"], peak=r["back"]["peak"],
                           peak_err=r["back"]["peak_err"],
                           count=r["back"]["count"], med=r["back"]["med"],
                           p90=r["back"]["p90"], span=r["back"]["span"],
                           density=r["back"]["density"])

        vr = ValidResult(candidate=c, right=right, back=back, fix_score=r["score"])
        if gt:
            vr.gt_x, vr.gt_y, vr.gt_yaw = gt
            vr.gt_ok = (vr.pos_err is not None and vr.pos_err < 0.25
                        and vr.yaw_err is not None and vr.yaw_err < 0.175)
        results.append(vr)

    # Select best among those with right_ok AND back_ok
    wall_ok = [r for r in results if r.right.ok and r.back.ok]
    if not wall_ok:
        return results  # all NO_SEL

    # Sort by fix_score descending
    wall_ok.sort(key=lambda r: -r.fix_score)
    top1, top2 = wall_ok[0], (wall_ok[1] if len(wall_ok) > 1 else None)

    # AMBIGUOUS check
    if top2 and (top1.fix_score - top2.fix_score) < MIN_SCORE_MARGIN:
        for r in results:
            r.reject_reason = f"AMBIGUOUS: top1={top1.fix_score:.1f} top2={top2.fix_score:.1f} margin={MIN_SCORE_MARGIN}"
        return results

    top1.selected = True
    return results


# ═══ Candidate acquisition ═══

def acquire_candidates_with_simlauncher(seed: int):
    import rospy, subprocess as sp
    sp.run(['pkill', '-f', 'rosmaster'], capture_output=True)
    sp.run(['pkill', '-f', 'roslaunch'], capture_output=True)
    time.sleep(5)
    sys.path.insert(0, '/root/kuavo_ws/src/craic_simulator/utils')
    sys.path.insert(0, '/root/kuavo_ws/src/craic_simulator/lib')
    from sim_launcher import SimLauncher
    launcher = SimLauncher(scene="scene1", seed=seed, robot_version=52)
    launcher.start(node_name=f"v3val_s{seed}", timeout=120)
    time.sleep(8)
    sys.path.insert(0, '/tmp/nav_test')
    import corner_localizer as cl
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    import tf2_ros, tf.transformations as tft

    xy_vox, xy3_filt, angles, ranges = cl.collect_base_link_cloud(n_frames=N_FRAMES)
    if xy_vox is None or len(xy_vox) < 30: return [], [], launcher
    lines = cl.detect_lines(xy_vox, xy3_filt)
    if len(lines) < 2: return [], [], launcher
    pairs, _ = cl.find_orthogonal_pairs(lines)
    if not pairs: return [], [], launcher
    all_raw = []
    for p in pairs:
        all_raw.extend(cl.generate_candidates_for_pair(p['li'], p['lj'], xy3_filt, angles, ranges))
    if not all_raw: return [], [], launcher
    all_raw.sort(key=lambda c: -c['score'])
    candidates = [Candidate(id=i, x=c['x'], y=c['y'], yaw=c['yaw'], source="corner", orig_score=c['score'])
                  for i, c in enumerate(all_raw[:20])]

    tf_buffer = tf2_ros.Buffer(); tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)
    tf_trans, tf_rot = None, None
    try:
        t = tf_buffer.lookup_transform('base_link', 'radar', rospy.Time(0), rospy.Duration(2.0))
        tf_trans = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        q = t.transform.rotation
        tf_rot = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    except: pass
    frames = []
    for _ in range(N_FRAMES):
        try:
            msg = rospy.wait_for_message('/lidar/points', PointCloud2, timeout=2.0)
            pts = np.array(list(pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)))
            if len(pts) == 0: continue
            fid = msg.header.frame_id
            if fid and fid != 'base_link' and tf_rot is not None:
                pts[:, :3] = pts[:, :3] @ tf_rot.T + tf_trans
            r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            pts = pts[(r > R_MIN) & (r < R_MAX)]
            if len(pts) > 0: frames.append(pts)
            rospy.sleep(0.1)
        except: continue
    return candidates, frames, launcher


# ═══ CLI ═══

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='0,7,13,31,42')
    ap.add_argument('--attempts', type=int, default=3)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    log_dir = '/tmp/nav_test/logs'
    os.makedirs(log_dir, exist_ok=True)

    COLS = ['seed','attempt','candidate_id','source','x','y','yaw_deg','orig_score',
            'right_peak','right_peak_err','right_count','right_ok',
            'back_peak','back_peak_err','back_count','back_ok',
            'fix_score','selected','reject_reason','gt_ok',
            'gt_x','gt_y','gt_yaw_deg','pos_err','yaw_err_deg']

    sys.path.insert(0, '/tmp/nav_test')
    import corner_localizer as cl_ref
    GT_POSES = getattr(cl_ref, 'GT_POSES', {})

    all_rows = []
    total_correct_selected = 0
    total_miss = 0      # CANDIDATE_MISS: no gt_ok candidate
    total_misselect = 0 # VALIDATOR_MISSELECT: gt_ok exists but selected wrong

    for seed in seeds:
        print(f"\n{'─'*50}\nSEED {seed}\n{'─'*50}")
        cands, frames, launcher = acquire_candidates_with_simlauncher(seed)
        gt = GT_POSES.get(seed)
        if gt: print(f"GT: ({gt[0]:.3f}, {gt[1]:.3f}, {math.degrees(gt[2]):.0f}°)")

        for a in range(args.attempts):
            if a > 0:
                import rospy
                from sensor_msgs.msg import PointCloud2
                import sensor_msgs.point_cloud2 as pc2
                import tf2_ros, tf.transformations as tft
                import corner_localizer as cl2

                xy2, xy32, ang2, ran2 = cl2.collect_base_link_cloud(n_frames=N_FRAMES)
                if xy2 is None or len(xy2) < 30: continue
                lines2 = cl2.detect_lines(xy2, xy32)
                if len(lines2) < 2: continue
                pairs2, _ = cl2.find_orthogonal_pairs(lines2)
                if not pairs2: continue
                allr2 = []
                for p in pairs2:
                    allr2.extend(cl2.generate_candidates_for_pair(p['li'], p['lj'], xy32, ang2, ran2))
                if not allr2: continue
                allr2.sort(key=lambda c: -c['score'])
                cands = [Candidate(id=i, x=c['x'], y=c['y'], yaw=c['yaw'], source="corner", orig_score=c['score'])
                         for i, c in enumerate(allr2[:20])]

                tf_buf = tf2_ros.Buffer(); tf_lis = tf2_ros.TransformListener(tf_buf)
                rospy.sleep(0.5)
                tf_trans, tf_rot = None, None
                try:
                    t = tf_buf.lookup_transform('base_link', 'radar', rospy.Time(0), rospy.Duration(2.0))
                    tf_trans = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                    q = t.transform.rotation
                    tf_rot = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
                except: pass
                frames = []
                for _ in range(N_FRAMES):
                    try:
                        msg = rospy.wait_for_message('/lidar/points', PointCloud2, timeout=2.0)
                        pts = np.array(list(pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)))
                        r = np.sqrt(pts[:,0]**2+pts[:,1]**2)
                        pts = pts[(r>0.35)&(r<20)]
                        if len(pts)>0: frames.append(pts)
                        rospy.sleep(0.1)
                    except: continue
                if len(frames) < 3: continue

            if len(frames) < 3 or not cands:
                print(f"  A{a+1} SKIP")
                continue

            results = validate_candidates_at_fix(cands, frames, gt)
            sel = [r for r in results if r.selected]
            gt_ok_cands = [r for r in results if r.gt_ok]
            n_gt_ok = len(gt_ok_cands)
            n_cands = len(cands)
            sel_id = sel[0].candidate.id if sel else None
            sel_correct = sel and sel[0].gt_ok

            # Classification
            if n_gt_ok == 0:
                tag = "CANDIDATE_MISS"
                total_miss += 1
            elif sel_correct:
                tag = "CORRECT"
                total_correct_selected += 1
            elif sel:
                tag = "VALIDATOR_MISSELECT"
                total_misselect += 1
            else:
                tag = "NO_SEL"
                total_misselect += 1  # gt_ok exists but validator didn't pick anything

            mark = "✓" if sel_correct else "✗"
            print(f"  A{a+1} {tag}  n_cands={n_cands} gt_ok={n_gt_ok}  "
                  f"sel={'#'+str(sel_id) if sel_id is not None else 'NONE'}  {mark}")

            for r in results:
                all_rows.append(r.row_dict(seed, a + 1))

            # Failure detail
            if not sel_correct:
                if sel:
                    s = sel[0]
                    print(f"    [SELECTED C{s.candidate.id}] pos_err={s.pos_err:.3f}m/{math.degrees(s.yaw_err):.1f}°  "
                          f"R(peak={s.right.peak:.3f} err={s.right.peak_err:.3f} n={s.right.count} ok={s.right.ok})  "
                          f"B(peak={s.back.peak:.3f} err={s.back.peak_err:.3f} n={s.back.count} ok={s.back.ok})  "
                          f"score={s.fix_score:.1f}")
                if n_gt_ok > 0:
                    best = min(gt_ok_cands, key=lambda r: r.pos_err)
                    print(f"    [BEST_GT C{best.candidate.id}] pos_err={best.pos_err:.3f}m/{math.degrees(best.yaw_err):.1f}°  "
                          f"R(peak={best.right.peak:.3f} err={best.right.peak_err:.3f} n={best.right.count} ok={best.right.ok})  "
                          f"B(peak={best.back.peak:.3f} err={best.back.peak_err:.3f} n={best.back.count} ok={best.back.ok})  "
                          f"score={best.fix_score:.1f}")
                elif not sel:
                    # Show top by score
                    top = max(results, key=lambda r: r.fix_score)
                    print(f"    [TOP C{top.candidate.id}] "
                          f"R(peak={top.right.peak:.3f} err={top.right.peak_err:.3f} n={top.right.count} ok={top.right.ok})  "
                          f"B(peak={top.back.peak:.3f} err={top.back.peak_err:.3f} n={top.back.count} ok={top.back.ok})  "
                          f"score={top.fix_score:.1f}")

        launcher.stop()
        time.sleep(5)

    # ═══ Summary ═══
    print("\n" + "=" * 70)
    print("SUMMARY (v3: geometric-first + AMBIGUOUS rejection)")
    print("=" * 70)
    print(f"total_correct_selected:   {total_correct_selected}")
    print(f"candidate_miss:           {total_miss}")
    print(f"validator_misselect:      {total_misselect}")
    n_attempts = total_correct_selected + total_miss + total_misselect
    print(f"total_attempts:           {n_attempts}")
    print(f"Target: correct≥13/15, miss+misselect minimal")

    # Write logs
    csv_path = os.path.join(log_dir, 'v3_validator.csv')
    jl_path = os.path.join(log_dir, 'v3_validator.jsonl')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)
    with open(jl_path, 'w') as f:
        for r in all_rows: f.write(json.dumps(r) + '\n')
    print(f"Logs: {jl_path}  ({len(all_rows)} rows)")
