#!/usr/bin/env python3
"""
fixed_pose_validator.py — FIX-point geometric candidate selector.

Core idea:
  At FIX point (5.01, 3.50, yaw=pi), wall positions are known:
    RIGHT wall (NORTH): y ≈ -1.00 ± 0.12  → normal_axis=1, span along axis=0 (x)
    BACK  wall (EAST):  x ≈ -1.24 ± 0.12  → normal_axis=0, span along axis=1 (y)

  For each candidate pose, transform 5-frame base_link LiDAR → world → FIX frame,
  then check if walls appear at expected positions.

Production: call with real corner_localizer top20 candidates.
Test mode: use --mock for standalone validation.

Usage:
  python3 fixed_pose_validator.py --mock --seeds 0,7,13,31,42 --attempts 3
"""

import sys, os, time, math, json, csv, argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import numpy as np

# ═══ Constants ═══
FIX_X = 5.01
FIX_Y = 3.50
FIX_YAW = math.pi              # facing west (180°)

NORTH_Y = 4.50                 # world Y of north wall
EAST_X = 6.25                  # world X of east wall

RIGHT_TARGET = -(NORTH_Y - FIX_Y)   # -1.00
BACK_TARGET  = -(EAST_X - FIX_X)    # -1.24

STRIP_HALF = 0.12
MIN_RIGHT_COUNT = 800
MIN_BACK_COUNT  = 500
MAX_MED_RESIDUAL = 0.06
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
    def yaw_deg(self):
        return math.degrees(self.yaw) % 360


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

    # GT — evaluation only
    gt_x: Optional[float] = None
    gt_y: Optional[float] = None
    gt_yaw: Optional[float] = None

    @property
    def pos_err(self):
        if self.gt_x is None: return None
        return math.hypot(self.candidate.x - self.gt_x,
                          self.candidate.y - self.gt_y)

    @property
    def yaw_err(self):
        if self.gt_yaw is None: return None
        d = (self.candidate.yaw - self.gt_yaw) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    def row_dict(self, seed, attempt):
        return {
            "seed": seed, "attempt": attempt,
            "candidate_id": self.candidate.id,
            "source": self.candidate.source,
            "x": round(self.candidate.x, 4),
            "y": round(self.candidate.y, 4),
            "yaw_deg": round(self.candidate.yaw_deg, 2),
            "orig_score": round(self.candidate.orig_score, 4),
            "right_count": self.right.count,
            "right_med": round(self.right.med_residual, 4),
            "right_p90": round(self.right.p90_residual, 4),
            "right_span": round(self.right.span, 3),
            "right_density": self.right.density_bins,
            "right_ok": self.right.ok,
            "back_count": self.back.count,
            "back_med": round(self.back.med_residual, 4),
            "back_p90": round(self.back.p90_residual, 4),
            "back_span": round(self.back.span, 3),
            "back_density": self.back.density_bins,
            "back_ok": self.back.ok,
            "fix_score": round(self.fix_score, 2),
            "selected": self.selected,
            "reject_reason": self.reject_reason,
            "gt_x": round(self.gt_x, 4) if self.gt_x else None,
            "gt_y": round(self.gt_y, 4) if self.gt_y else None,
            "gt_yaw_deg": round(math.degrees(self.gt_yaw), 2) if self.gt_yaw else None,
            "pos_err": round(self.pos_err, 4) if self.pos_err is not None else None,
            "yaw_err_deg": round(math.degrees(self.yaw_err), 2) if self.yaw_err is not None else None,
        }

# ═══ Point cloud ═══

def voxel_downsample(points: np.ndarray, size: float = VOXEL_SIZE) -> np.ndarray:
    """Deterministic voxel centroid."""
    if len(points) < 2: return points
    vox = np.floor(points[:, :3] / size).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return points[idx]


def collect_base_link_frames(n: int = N_FRAMES) -> List[np.ndarray]:
    """Collect N pointcloud frames from /lidar/points, TF to base_link.

    Same approach as corner_localizer: assumes lidar_frame=base_link from launch.
    For safety, also tries tf2_ros to transform if frame_id differs.
    """
    import rospy
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    import tf2_ros

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.3)

    frames = []
    for _ in range(n):
        try:
            msg = rospy.wait_for_message('/lidar/points', PointCloud2, timeout=2.0)
            field_names = ('x', 'y', 'z')
            pts = np.array(list(pc2.read_points(msg, field_names=field_names,
                                                skip_nans=True)))
            if len(pts) == 0:
                continue

            # If frame_id != "base_link", transform via TF
            if msg.header.frame_id != 'base_link' and msg.header.frame_id:
                try:
                    t = tf_buffer.lookup_transform('base_link', msg.header.frame_id,
                                                   msg.header.stamp, rospy.Duration(1.0))
                    dx, dy, dz = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
                    q = t.transform.rotation
                    import tf.transformations as tft
                    R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
                    pts[:, :3] = pts[:, :3] @ R.T + np.array([dx, dy, dz])
                except Exception:
                    pass  # frame_id already base_link, or transform unavailable

            r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            pts = pts[(r > R_MIN) & (r < R_MAX)]
            if len(pts) > 0:
                frames.append(pts)
            rospy.sleep(0.1)
        except Exception:
            continue
    return frames


# ═══ Transform ═══

def transform_to_world(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Rotate base_link points by yaw, then translate by (x,y)."""
    if len(points) == 0: return points
    c, s = math.cos(yaw), math.sin(yaw)
    out = points.copy()
    out[:, 0] = points[:, 0] * c - points[:, 1] * s + x
    out[:, 1] = points[:, 0] * s + points[:, 1] * c + y
    return out


def world_to_fix(points_world: np.ndarray) -> np.ndarray:
    """Translate world points to FIX origin, rotate by -FIX_YAW into FIX base_link."""
    if len(points_world) == 0: return points_world
    out = points_world.copy()
    out[:, 0] -= FIX_X
    out[:, 1] -= FIX_Y
    c, s = math.cos(-FIX_YAW), math.sin(-FIX_YAW)
    x = out[:, 0] * c - out[:, 1] * s
    y = out[:, 0] * s + out[:, 1] * c
    out[:, 0], out[:, 1] = x, y
    return out


def points_to_fix_frame(points_base: np.ndarray, cand: Candidate) -> np.ndarray:
    """base_link → world → FIX base_link."""
    p_world = transform_to_world(points_base, cand.x, cand.y, cand.yaw)
    return world_to_fix(p_world)


# ═══ Strip evaluation (FIXED axis logic) ═══

def compute_strip(points_fix: np.ndarray,
                  target: float,
                  normal_axis: int,      # which axis to check distance on
                  along_axis: int,       # which axis to measure span along
                  min_count: int,
                  min_span: float,
                  strip_half: float = STRIP_HALF) -> StripMetrics:
    """Compute wall strip metrics in FIX frame.

    normal_axis: axis perpendicular to wall (0=x for EAST wall, 1=y for NORTH wall)
    along_axis:  axis parallel to wall     (1=y for EAST wall, 0=x for NORTH wall)

    Example:
      RIGHT (NORTH) wall: normal_axis=1 (check y), along_axis=0 (span along x)
      BACK  (EAST)  wall: normal_axis=0 (check x), along_axis=1 (span along y)
    """
    sm = StripMetrics()
    residuals = np.abs(points_fix[:, normal_axis] - target)
    in_strip = residuals < strip_half
    sm.count = int(np.sum(in_strip))

    if sm.count < 5:
        return sm

    strip_pts = points_fix[in_strip]
    strip_res = residuals[in_strip]
    sm.med_residual = float(np.median(strip_res))
    sm.p90_residual = float(np.percentile(strip_res, 90))
    sm.span = float(np.max(strip_pts[:, along_axis]) - np.min(strip_pts[:, along_axis]))

    # Density: 0.1m bins, count bins with ≥10 points
    bin_size = 0.1
    n_bins = max(1, int(sm.span / bin_size))
    offset = np.min(strip_pts[:, along_axis])
    bins = np.floor((strip_pts[:, along_axis] - offset) / bin_size).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    bin_counts = np.bincount(bins, minlength=n_bins)
    sm.density_bins = int(np.sum(bin_counts >= 10))

    sm.ok = (sm.count >= min_count
             and sm.med_residual <= MAX_MED_RESIDUAL
             and sm.span >= min_span
             and sm.density_bins >= MIN_DENSITY_BINS)
    return sm


# ═══ Scoring ═══

def score_candidate(right: StripMetrics, back: StripMetrics) -> Tuple[float, str]:
    """Return (score, reject_reason). Score=0 if not passing both walls."""
    if not right.ok or not back.ok:
        parts = []
        if not right.ok:
            parts.append(f"RIGHT_FAIL(n={right.count}/{MIN_RIGHT_COUNT},"
                         f" med={right.med_residual:.3f}<={MAX_MED_RESIDUAL},"
                         f" sp={right.span:.2f}>={MIN_RIGHT_SPAN},"
                         f" dens={right.density_bins}<{MIN_DENSITY_BINS})")
        if not back.ok:
            parts.append(f"BACK_FAIL(n={back.count}/{MIN_BACK_COUNT},"
                         f" med={back.med_residual:.3f}<={MAX_MED_RESIDUAL},"
                         f" sp={back.span:.2f}>={MIN_BACK_SPAN},"
                         f" dens={back.density_bins}<{MIN_DENSITY_BINS})")
        return 0.0, "; ".join(parts)

    s = 0.0
    s += 80.0  # right pass
    s += 80.0  # back pass
    s += min(right.count, 3000) * 0.01
    s += min(back.count, 2500) * 0.01
    s += min(right.span, 2.0) * 10.0
    s += min(back.span, 2.0) * 10.0
    s -= 300.0 * right.med_residual
    s -= 300.0 * back.med_residual
    s -= 100.0 * max(0.0, right.p90_residual - 0.12)
    s -= 100.0 * max(0.0, back.p90_residual - 0.12)
    return s, ""


# ═══ Validator ═══

def validate(candidates: List[Candidate],
             points_frames: List[np.ndarray],
             gt: Optional[Tuple[float, float, float]] = None) -> List[ValidResult]:
    """Validate all candidates against FIX geometry.

    Returns: list of ValidResult, with .selected=True on best passing candidate.
    If no candidate passes both right_ok AND back_ok, .selected=False on all.
    GT is used ONLY for evaluation logging, never for selection.
    """
    if not points_frames:
        return []

    merged = np.vstack(points_frames)
    merged = voxel_downsample(merged)

    results = []
    for c in candidates:
        p_fix = points_to_fix_frame(merged, c)

        # RIGHT wall (NORTH): check y (normal_axis=1), span along x (along_axis=0)
        right = compute_strip(p_fix, RIGHT_TARGET,
                              normal_axis=1, along_axis=0,
                              min_count=MIN_RIGHT_COUNT, min_span=MIN_RIGHT_SPAN)

        # BACK wall (EAST): check x (normal_axis=0), span along y (along_axis=1)
        back = compute_strip(p_fix, BACK_TARGET,
                             normal_axis=0, along_axis=1,
                             min_count=MIN_BACK_COUNT, min_span=MIN_BACK_SPAN)

        score, reason = score_candidate(right, back)
        r = ValidResult(candidate=c, right=right, back=back,
                        fix_score=score, reject_reason=reason)
        if gt:
            r.gt_x, r.gt_y, r.gt_yaw = gt
        results.append(r)

    valid = [r for r in results if r.right.ok and r.back.ok]
    if valid:
        max(valid, key=lambda r: r.fix_score).selected = True

    return results


# ═══ Test harness ═══

def run_one_test(seed: int, attempt: int, mock: bool = True,
                 candidate_fn: Optional[Callable] = None) -> Tuple[List[ValidResult], Optional[tuple]]:
    """Run one validation test.

    If mock=True: use _mock_candidates + synthetic GT.
    If mock=False: use candidate_fn() for real corner_localizer output.
    Returns (results, gt_tuple).
    """
    frames = collect_base_link_frames(N_FRAMES)
    if len(frames) < 2:
        print(f"  seed={seed} attempt={attempt}: only {len(frames)} frames, SKIP")
        return [], None

    if mock:
        cands = _mock_candidates(seed, attempt)
        gt = (FIX_X + (seed % 13 - 6) * 0.02,
              FIX_Y + (seed % 7 - 3) * 0.02,
              FIX_YAW + math.radians((seed % 5 - 2) * 0.5))
    else:
        if candidate_fn is None:
            raise ValueError("Production mode requires candidate_fn")
        cands = candidate_fn(seed, attempt)
        gt = None  # No GT in production; eval to be done separately

    return validate(cands, frames, gt), gt


def _mock_candidates(seed: int, attempt: int) -> List[Candidate]:
    """Mock candidates for standalone axis-verification testing."""
    np.random.seed(seed * 100 + attempt)
    return [
        Candidate(0, FIX_X + np.random.normal(0, 0.03), FIX_Y + np.random.normal(0, 0.03),
                  FIX_YAW + np.random.normal(0, 0.03), source="corner", orig_score=0.92),
        Candidate(1, FIX_X + np.random.normal(0, 0.03), FIX_Y + np.random.normal(0, 0.03),
                  (FIX_YAW + math.pi/2) % (2*math.pi) + np.random.normal(0, 0.05),
                  source="corner", orig_score=0.88),
        Candidate(2, FIX_X + np.random.normal(0, 0.8), FIX_Y + np.random.normal(0, 0.8),
                  np.random.uniform(0, 2*math.pi), source="corner", orig_score=0.45),
        Candidate(3, FIX_X + np.random.normal(0, 0.6), FIX_Y + np.random.normal(0, 0.6),
                  FIX_YAW + math.pi + np.random.normal(0, 0.1),
                  source="corner", orig_score=0.40),
    ]


# ═══ CLI ═══

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mock', action='store_true', help='Use mock candidates (for axis verification)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', type=str, default=None)
    ap.add_argument('--attempts', type=int, default=3)
    args = ap.parse_args()

    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]

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

    for seed in seeds:
        for a in range(args.attempts):
            results, gt = run_one_test(seed, a, mock=args.mock)
            if not results:
                continue
            total += 1
            sel = [r for r in results if r.selected]
            print(f"\n── seed={seed} attempt={a}  {len(results)} candidates  "
                  f"selected={'#'+str(sel[0].candidate.id) if sel else 'NONE'} ──")

            for r in results:
                row = r.row_dict(seed, a)
                all_rows.append(row)
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
                print(f"  ⚠ NO CANDIDATE PASSED")  # handled by --mock; production will also log

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    with open(jsonl_path, 'w') as f:
        for row in all_rows:
            f.write(json.dumps(row) + '\n')

    print(f"\n{'='*60}")
    if total:
        print(f"  ACCURACY: {correct}/{total}  (pos_err<0.25m & yaw_err<10°)")
    else:
        print(f"  No data — is /lidar/points publishing?")
    print(f"  Logs: {csv_path}  {jsonl_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
