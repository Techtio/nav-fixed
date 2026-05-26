#!/usr/bin/env python3
"""
fixed_pose_validator.py — Production candidate selector using FIX-point geometry.

Calls corner_localizer for top20 candidates, then validates each with
FIX-frame wall positions (RIGHT y=-1.00, BACK x=-1.24).

Usage:
  python3 fixed_pose_validator.py --seed 0 --attempts 3
  python3 fixed_pose_validator.py --seeds 0,7,13,31,42 --attempts 3
"""

import sys, os, time, math, json, csv, argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# ═══ Constants ═══
FIX_X = 5.01
FIX_Y = 3.50
FIX_YAW = math.pi

NORTH_Y = 4.50
EAST_X = 6.25
RIGHT_TARGET = -(NORTH_Y - FIX_Y)   # -1.00
BACK_TARGET  = -(EAST_X - FIX_X)    # -1.24

STRIP_HALF = 0.12
MIN_RIGHT_COUNT = 800
MIN_BACK_COUNT  = 500
MAX_MED_RESIDUAL = 0.05
MIN_RIGHT_SPAN = 0.6
MIN_BACK_SPAN  = 0.5
MIN_DENSITY_BINS = 3

N_FRAMES = 5
VOXEL_SIZE = 0.03
R_MIN = 0.35
R_MAX = 20.0

# ═══ Data ═══

@dataclass
class Candidate:
    id: int
    x: float
    y: float
    yaw: float
    source: str = ""
    orig_score: float = 0.0

    @property
    def yaw_deg(self): return math.degrees(self.yaw) % 360


@dataclass
class StripMetrics:
    count: int = 0
    med_residual: float = 999.0
    p90_residual: float = 999.0
    span: float = 0.0
    density_bins: int = 0
    ok: bool = False


@dataclass
class ValidResult:
    candidate: Candidate
    right: StripMetrics = field(default_factory=StripMetrics)
    back: StripMetrics = field(default_factory=StripMetrics)
    fix_score: float = 0.0
    selected: bool = False
    reject_reason: str = ""
    gt_x: Optional[float] = None
    gt_y: Optional[float] = None
    gt_yaw: Optional[float] = None

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
            "right_count": self.right.count,
            "right_med": round(self.right.med_residual, 4),
            "right_p90": round(self.right.p90_residual, 4),
            "right_span": round(self.right.span, 3),
            "right_density": self.right.density_bins, "right_ok": self.right.ok,
            "back_count": self.back.count,
            "back_med": round(self.back.med_residual, 4),
            "back_p90": round(self.back.p90_residual, 4),
            "back_span": round(self.back.span, 3),
            "back_density": self.back.density_bins, "back_ok": self.back.ok,
            "fix_score": round(self.fix_score, 2), "selected": self.selected,
            "reject_reason": self.reject_reason,
        }
        if self.gt_x is not None:
            r.update({"gt_x": round(self.gt_x, 4), "gt_y": round(self.gt_y, 4),
                      "gt_yaw_deg": round(math.degrees(self.gt_yaw), 2),
                      "pos_err": round(self.pos_err, 4) if self.pos_err else None,
                      "yaw_err_deg": round(math.degrees(self.yaw_err), 2) if self.yaw_err else None})
        return r


# ═══ Point cloud (reuses corner_localizer's approach) ═══

def collect_base_link_frames(n: int = N_FRAMES) -> List[np.ndarray]:
    """Collect N /lidar/points frames, each TF'd to base_link.

    Same approach as corner_localizer: uses tf2_ros to transform from
    radar → base_link via lookup_transform.
    """
    import rospy
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    import tf2_ros, tf2_geometry_msgs
    import tf.transformations as tft

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)

    # Cache TF for radar → base_link
    tf_trans, tf_rot = None, None
    try:
        t = tf_buffer.lookup_transform('base_link', 'radar', rospy.Time(0), rospy.Duration(2.0))
        tf_trans = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        q = t.transform.rotation
        tf_rot = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    except Exception as e:
        pass  # lidar_frame may already be base_link

    frames = []
    for _ in range(n):
        try:
            msg = rospy.wait_for_message('/lidar/points', PointCloud2, timeout=2.0)
            pts = np.array(list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)))
            if len(pts) == 0: continue

            # Apply TF if frame_id ≠ base_link AND we have a valid transform
            fid = msg.header.frame_id
            if fid and fid != 'base_link' and tf_rot is not None and tf_trans is not None:
                pts[:, :3] = pts[:, :3] @ tf_rot.T + tf_trans
            # Also try direct transform if fid varies
            elif fid and fid != 'base_link':
                try:
                    t2 = tf_buffer.lookup_transform('base_link', fid, msg.header.stamp, rospy.Duration(1.0))
                    dx = t2.transform.translation.x; dy = t2.transform.translation.y; dz = t2.transform.translation.z
                    q2 = t2.transform.rotation
                    R2 = tft.quaternion_matrix([q2.x, q2.y, q2.z, q2.w])[:3, :3]
                    pts[:, :3] = pts[:, :3] @ R2.T + np.array([dx, dy, dz])
                except:
                    pass

            r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            pts = pts[(r > R_MIN) & (r < R_MAX)]
            if len(pts) > 0: frames.append(pts)
            rospy.sleep(0.1)
        except: continue
    return frames


# ═══ Transform ═══

def transform_to_world(pts: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    if len(pts) == 0: return pts
    c, s = math.cos(yaw), math.sin(yaw)
    out = pts.copy()
    out[:, 0] = pts[:, 0] * c - pts[:, 1] * s + x
    out[:, 1] = pts[:, 0] * s + pts[:, 1] * c + y
    return out


def world_to_fix(pts_w: np.ndarray) -> np.ndarray:
    if len(pts_w) == 0: return pts_w
    out = pts_w.copy()
    out[:, 0] -= FIX_X; out[:, 1] -= FIX_Y
    c, s = math.cos(-FIX_YAW), math.sin(-FIX_YAW)
    x = out[:, 0] * c - out[:, 1] * s
    y = out[:, 0] * s + out[:, 1] * c
    out[:, 0], out[:, 1] = x, y
    return out


def voxel_downsample(pts: np.ndarray, size: float = VOXEL_SIZE) -> np.ndarray:
    if len(pts) < 2: return pts
    vox = np.floor(pts[:, :3] / size).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return pts[idx]


# ═══ Strip ═══

def compute_strip(pts_fix: np.ndarray, target: float,
                  normal_axis: int, along_axis: int,
                  min_count: int, min_span: float) -> StripMetrics:
    sm = StripMetrics()
    residuals = np.abs(pts_fix[:, normal_axis] - target)
    in_strip = residuals < STRIP_HALF
    sm.count = int(np.sum(in_strip))
    if sm.count < 5: return sm

    strip_pts = pts_fix[in_strip]
    strip_res = residuals[in_strip]
    sm.med_residual = float(np.median(strip_res))
    sm.p90_residual = float(np.percentile(strip_res, 90))
    sm.span = float(np.max(strip_pts[:, along_axis]) - np.min(strip_pts[:, along_axis]))

    bin_size = 0.1
    n_bins = max(1, int(sm.span / bin_size))
    off = np.min(strip_pts[:, along_axis])
    bins = np.floor((strip_pts[:, along_axis] - off) / bin_size).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    bin_counts = np.bincount(bins, minlength=n_bins)
    sm.density_bins = int(np.sum(bin_counts >= 10))
    sm.ok = (sm.count >= min_count and sm.med_residual <= MAX_MED_RESIDUAL
             and sm.span >= min_span and sm.density_bins >= MIN_DENSITY_BINS)
    return sm


# ═══ Score ═══

def score_candidate(right: StripMetrics, back: StripMetrics) -> Tuple[float, str]:
    if not right.ok or not back.ok:
        parts = []
        if not right.ok: parts.append(f"RIGHT(n={right.count} med={right.med_residual:.3f} sp={right.span:.2f})")
        if not back.ok: parts.append(f"BACK(n={back.count} med={back.med_residual:.3f} sp={back.span:.2f})")
        return 0.0, "; ".join(parts)
    s = 160.0
    s += min(right.count, 3000) * 0.01 + min(back.count, 2500) * 0.01
    s += min(right.span, 2.0) * 10.0 + min(back.span, 2.0) * 10.0
    s -= 300.0 * right.med_residual + 300.0 * back.med_residual
    s -= 100.0 * max(0.0, right.p90_residual - 0.12)
    s -= 100.0 * max(0.0, back.p90_residual - 0.12)
    return s, ""


# ═══ Real candidate acquisition ═══

def get_real_candidates(seed: int, attempt: int):
    """Call corner_localizer detection → return (candidates, pointcloud_frames).
    Pointcloud frames come from the same /lidar/points source, TF'd to base_link.
    """
    sys.path.insert(0, '/tmp/nav_test')
    sys.path.insert(0, '/root/kuavo_ws/src/craic_simulator/utils')
    sys.path.insert(0, '/root/kuavo_ws/src/craic_simulator/lib')
    import corner_localizer as cl

    # 1. Collect cloud just like corner_localizer does
    xy_vox, xy3_filt, angles, ranges = cl.collect_base_link_cloud(n_frames=N_FRAMES)
    if xy_vox is None or len(xy_vox) < 30:
        return [], []

    # 2. Detect lines and generate all candidates (same as corner_localizer)
    lines = cl.detect_lines(xy_vox, xy3_filt)
    if len(lines) < 2: return [], []
    pairs, _ = cl.find_orthogonal_pairs(lines)
    if not pairs: return [], []

    all_cands_raw = []
    for p in pairs:
        all_cands_raw.extend(cl.generate_candidates_for_pair(p['li'], p['lj'], xy3_filt, angles, ranges))
    if not all_cands_raw: return [], []

    all_cands_raw.sort(key=lambda c: -c['score'])
    top20 = all_cands_raw[:20]

    # Convert to our Candidate type
    candidates = []
    for i, c in enumerate(top20):
        candidates.append(Candidate(
            id=i, x=c['x'], y=c['y'], yaw=c['yaw'],
            source="corner", orig_score=c['score']
        ))

    # Re-collect raw pointcloud frames (same source, for FIX validation)
    import rospy
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    import tf2_ros, tf.transformations as tft
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.3)

    # Cache radar→base_link TF
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
            pts = np.array(list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)))
            if len(pts) == 0: continue
            fid = msg.header.frame_id
            if fid and fid != 'base_link' and tf_rot is not None:
                pts[:, :3] = pts[:, :3] @ tf_rot.T + tf_trans
            r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            pts = pts[(r > R_MIN) & (r < R_MAX)]
            if len(pts) > 0: frames.append(pts)
            rospy.sleep(0.1)
        except: continue
    return candidates, frames


# ═══ Validate ═══

# ═══ Main validator entry point ═══

def validate_candidates_at_fix(
    candidates: List[Candidate],
    points_frames: List[np.ndarray],
    gt: Optional[Tuple[float, float, float]] = None
) -> List[ValidResult]:
    if not frames: return []
    merged = np.vstack(frames)
    merged = voxel_downsample(merged)

    results = []
    for c in candidates:
        p_world = transform_to_world(merged, c.x, c.y, c.yaw)
        p_fix = world_to_fix(p_world)
        right = compute_strip(p_fix, RIGHT_TARGET, normal_axis=1, along_axis=0,
                              min_count=MIN_RIGHT_COUNT, min_span=MIN_RIGHT_SPAN)
        back  = compute_strip(p_fix, BACK_TARGET, normal_axis=0, along_axis=1,
                              min_count=MIN_BACK_COUNT, min_span=MIN_BACK_SPAN)
        score, reason = score_candidate(right, back)
        r = ValidResult(candidate=c, right=right, back=back, fix_score=score, reject_reason=reason)
        if gt: r.gt_x, r.gt_y, r.gt_yaw = gt
        results.append(r)

    valid = [r for r in results if r.right.ok and r.back.ok]
    if valid:
        max(valid, key=lambda r: r.fix_score).selected = True
    return results


# ═══ CLI ═══

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', type=str, default=None)
    ap.add_argument('--attempts', type=int, default=3)
    args = ap.parse_args()

    seeds = [args.seed]
    if args.seeds: seeds = [int(s.strip()) for s in args.seeds.split(',')]

    log_dir = '/tmp/nav_test/logs'
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'fix_validator.csv')
    jsonl_path = os.path.join(log_dir, 'fix_validator.jsonl')

    cols = ['seed', 'attempt', 'candidate_id', 'source', 'x', 'y', 'yaw_deg',
            'orig_score', 'right_count', 'right_med', 'right_p90', 'right_span',
            'right_density', 'right_ok', 'back_count', 'back_med', 'back_p90',
            'back_span', 'back_density', 'back_ok', 'fix_score', 'selected',
            'reject_reason', 'gt_x', 'gt_y', 'gt_yaw_deg', 'pos_err', 'yaw_err_deg']

    all_rows, correct, total = [], 0, 0

    # Import once for GT_POSES
    sys.path.insert(0, '/tmp/nav_test')
    import corner_localizer as cl_ref
    GT_POSES = getattr(cl_ref, 'GT_POSES', {})

    for seed in seeds:
        for a in range(args.attempts):
            cands, frames = get_real_candidates(seed, a + 1)
            if len(frames) < 3 or not cands:
                print(f"  seed={seed} attempt={a+1}: frames={len(frames)} cands={len(cands)} SKIP")
                continue

            gt = GT_POSES.get(seed, None)
            results = validate_candidates_at_fix(cands, frames, gt)
            total += 1

            sel = [r for r in results if r.selected]
            print(f"\n── seed={seed} attempt={a+1}  {len(results)} candidates  "
                  f"selected={'#'+str(sel[0].candidate.id) if sel else 'NONE'} ──")

            for r in results:
                all_rows.append(r.row_dict(seed, a + 1))
                mark = " ★" if r.selected else ""
                perr = f"pos={r.pos_err:.3f}" if r.pos_err else ""
                yerr = f"yaw={math.degrees(r.yaw_err):.1f}°" if r.yaw_err else ""
                print(f"  [{r.candidate.id}] R(n={r.right.count} med={r.right.med_residual:.3f} "
                      f"sp={r.right.span:.2f} ok={r.right.ok})  "
                      f"B(n={r.back.count} med={r.back.med_residual:.3f} "
                      f"sp={r.back.span:.2f} ok={r.back.ok})  "
                      f"score={r.fix_score:.1f}  {perr} {yerr}{mark}")

            if sel:
                best = sel[0]
                if best.pos_err is not None and best.yaw_err is not None:
                    if best.pos_err < 0.25 and best.yaw_err < 0.17:
                        correct += 1
            else:
                print(f"  ⚠ NO CANDIDATE PASSED both right_ok AND back_ok — FAIL")
                print(f"  Failure analysis:")
                for r in results:
                    print(f"    [{r.candidate.id}] right_ok={r.right.ok} back_ok={r.back.ok} "
                          f"R(n={r.right.count} med={r.right.med_residual:.3f} sp={r.right.span:.2f}) "
                          f"B(n={r.back.count} med={r.back.med_residual:.3f} sp={r.back.span:.2f})")

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)
    with open(jsonl_path, 'w') as f:
        for row in all_rows: f.write(json.dumps(row) + '\n')

    print(f"\n{'='*60}")
    if total:
        print(f"  ACCURACY: {correct}/{total}  (pos_err<0.25m & yaw_err<10°)")
    print(f"  Total tests: {total}  Logs: {csv_path}  {jsonl_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
