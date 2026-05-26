#!/usr/bin/env python3
"""
fixed_pose_validator.py — FIX-point geometric candidate selector.

Algorithm:
  For each candidate pose from corner_localizer:
    1. Transform 5-frame LiDAR pointcloud to FIX base_link frame
    2. In FIX frame, RIGHT wall (NORTH) must be at y ≈ -1.00
       BACK wall (EAST) must be at x ≈ -1.24
    3. Score each candidate; select only those passing both wall checks.

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
FIX_YAW = math.pi  # facing west

NORTH_Y = 4.50
EAST_X = 6.25
RIGHT_TARGET = -(NORTH_Y - FIX_Y)   # -1.00
BACK_TARGET  = -(EAST_X - FIX_X)    # -1.24

STRIP_HALF = 0.12              # |dist| < STRIP_HALF

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

# ═══ Data structures ═══

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

    # GT — evaluation only, NOT used for selection
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

    def row_dict(self, seed: int, attempt: int) -> dict:
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

# ═══ Point cloud helpers ═══

def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    """Deterministic voxel centroid downsampling (no randomness)."""
    if len(points) < 2:
        return points
    vox = np.floor(points[:, :3] / size).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return points[idx]


def transform_to_world(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Rotate by yaw then translate by (x,y). Modifies columns 0,1."""
    if len(points) == 0:
        return points
    c, s = math.cos(yaw), math.sin(yaw)
    out = points.copy()
    out[:, 0] = points[:, 0] * c - points[:, 1] * s + x
    out[:, 1] = points[:, 0] * s + points[:, 1] * c + y
    return out


def world_to_fix(points_world: np.ndarray) -> np.ndarray:
    """Translate to FIX origin, then rotate by -FIX_YAW into FIX base_link frame."""
    out = points_world.copy()
    out[:, 0] -= FIX_X
    out[:, 1] -= FIX_Y
    c, s = math.cos(-FIX_YAW), math.sin(-FIX_YAW)
    x = out[:, 0] * c - out[:, 1] * s
    y = out[:, 0] * s + out[:, 1] * c
    out[:, 0] = x
    out[:, 1] = y
    return out


def load_pointcloud_frames(n_frames: int = N_FRAMES) -> List[np.ndarray]:
    """Collect N_FRAMES LiDAR pointclouds from /lidar/points in base_link."""
    import rospy
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    frames = []
    for _ in range(n_frames):
        try:
            msg = rospy.wait_for_message('/lidar/points', PointCloud2, timeout=2.0)
            pts = np.array(list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)))
            r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            pts = pts[(r > R_MIN) & (r < R_MAX)]
            frames.append(pts)
            rospy.sleep(0.1)
        except Exception:
            continue
    return frames


# ═══ Strip evaluation ═══

def compute_strip(points_fix: np.ndarray, target: float, strip_axis: int,
                  min_count: int, min_span: float) -> StripMetrics:
    """Compute wall-strip metrics in FIX frame.

    strip_axis=0 → check |y - target| < STRIP_HALF, span along x (RIGHT wall)
    strip_axis=1 → check |x - target| < STRIP_HALF, span along y (BACK wall)
    """
    sm = StripMetrics()
    along_axis = 1 - strip_axis  # if checking x, span along y; if checking y, span along x
    residuals = np.abs(points_fix[:, strip_axis] - target)
    in_strip = residuals < STRIP_HALF
    sm.count = int(np.sum(in_strip))
    if sm.count < 5:
        return sm

    strip_pts = points_fix[in_strip]
    strip_res = residuals[in_strip]
    sm.med_residual = float(np.median(strip_res))
    sm.p90_residual = float(np.percentile(strip_res, 90))
    sm.span = float(np.max(strip_pts[:, along_axis]) - np.min(strip_pts[:, along_axis]))

    # Density: count 0.1m bins with ≥10 points
    bin_size = 0.1
    if sm.span > 0:
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
    reasons = []
    if not right.ok:
        reasons.append(f"RIGHT_FAIL(count={right.count}<{MIN_RIGHT_COUNT} or "
                       f"med={right.med_residual:.3f}>{MAX_MED_RESIDUAL} or "
                       f"span={right.span:.2f}<{MIN_RIGHT_SPAN})")
    if not back.ok:
        reasons.append(f"BACK_FAIL(count={back.count}<{MIN_BACK_COUNT} or "
                       f"med={back.med_residual:.3f}>{MAX_MED_RESIDUAL} or "
                       f"span={back.span:.2f}<{MIN_BACK_SPAN})")

    if reasons:
        return 0.0, "; ".join(reasons)

    s = 0.0
    s += 80.0  # right_ok bonus
    s += 80.0  # back_ok bonus
    s += min(right.count, 3000) * 0.01
    s += min(back.count, 2500) * 0.01
    s += min(right.span, 2.0) * 10.0
    s += min(back.span, 2.0) * 10.0
    s -= 300.0 * right.med_residual
    s -= 300.0 * back.med_residual
    s -= 100.0 * max(0.0, right.p90_residual - 0.12)
    s -= 100.0 * max(0.0, back.p90_residual - 0.12)
    return s, ""


# ═══ Main validator ═══

def validate(candidates: List[Candidate],
             points_frames: List[np.ndarray],
             gt: Optional[Tuple[float, float, float]] = None) -> List[ValidResult]:
    """Run FIX geometry validation on all candidates. Returns list with .selected set on best valid candidate, or all False if none pass."""
    if not points_frames:
        return []
    merged = np.vstack(points_frames)
    merged = voxel_downsample(merged, VOXEL_SIZE)

    results = []
    for c in candidates:
        p_world = transform_to_world(merged, c.x, c.y, c.yaw)
        p_fix = world_to_fix(p_world)

        right = compute_strip(p_fix, RIGHT_TARGET, strip_axis=0,
                              min_count=MIN_RIGHT_COUNT, min_span=MIN_RIGHT_SPAN)
        back  = compute_strip(p_fix, BACK_TARGET, strip_axis=1,
                              min_count=MIN_BACK_COUNT, min_span=MIN_BACK_SPAN)
        score, reason = score_candidate(right, back)
        r = ValidResult(candidate=c, right=right, back=back,
                        fix_score=score, reject_reason=reason)
        if gt:
            r.gt_x, r.gt_y, r.gt_yaw = gt
        results.append(r)

    # Select: only candidates with BOTH right_ok AND back_ok
    valid = [r for r in results if r.right.ok and r.back.ok]
    if valid:
        best = max(valid, key=lambda r: r.fix_score)
        best.selected = True

    return results


# ═══ Test harness ═══

def run_one_test(seed: int, attempt: int) -> List[ValidResult]:
    """Collect frames, get mock candidates, validate."""
    frames = load_pointcloud_frames(N_FRAMES)
    if len(frames) < 2:
        print(f"  seed={seed} attempt={attempt}: only {len(frames)} frames, SKIP")
        return []

    cands = _mock_candidates(seed, attempt)
    gt = (FIX_X + (seed % 13 - 6) * 0.02,
          FIX_Y + (seed % 7 - 3) * 0.02,
          FIX_YAW + math.radians((seed % 5 - 2) * 0.5))  # eval-only GT
    return validate(cands, frames, gt)


def _mock_candidates(seed: int, attempt: int) -> List[Candidate]:
    """Generate mock candidates for standalone testing.
    Production: replace with corner_localizer.query_candidates().
    """
    np.random.seed(seed * 100 + attempt)
    cands = []
    # Correct
    cands.append(Candidate(0, FIX_X + np.random.normal(0, 0.03), FIX_Y + np.random.normal(0, 0.03),
                           FIX_YAW + np.random.normal(0, 0.03), source="corner", orig_score=0.92))
    # Swapped
    cands.append(Candidate(1, FIX_X + np.random.normal(0, 0.03), FIX_Y + np.random.normal(0, 0.03),
                           (FIX_YAW + math.pi/2) % (2*math.pi) + np.random.normal(0, 0.05),
                           source="corner", orig_score=0.88))
    # Noise
    for i in range(2, 5):
        cands.append(Candidate(i, FIX_X + np.random.normal(0, 0.8), FIX_Y + np.random.normal(0, 0.8),
                               np.random.uniform(0, 2*math.pi), source="corner", orig_score=0.45))
    return cands


# ═══ CLI ═══

def main():
    ap = argparse.ArgumentParser()
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

    # CSV schema
    cols = ['seed', 'attempt', 'candidate_id', 'source', 'x', 'y', 'yaw_deg',
            'orig_score', 'right_count', 'right_med', 'right_p90', 'right_span',
            'right_density', 'right_ok', 'back_count', 'back_med', 'back_p90',
            'back_span', 'back_density', 'back_ok', 'fix_score', 'selected',
            'reject_reason', 'gt_x', 'gt_y', 'gt_yaw_deg', 'pos_err', 'yaw_err_deg']

    all_rows, correct, total, passed = [], 0, 0, 0

    for seed in seeds:
        for a in range(args.attempts):
            results = run_one_test(seed, a)
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
                print(f"  [{r.candidate.id}] R(count={r.right.count} med={r.right.med_residual:.3f} "
                      f"span={r.right.span:.2f} ok={r.right.ok})  "
                      f"B(count={r.back.count} med={r.back.med_residual:.3f} "
                      f"span={r.back.span:.2f} ok={r.back.ok})  "
                      f"score={r.fix_score:.1f}  {perr} {yerr}{mark}")

            if sel:
                best = sel[0]
                if best.pos_err is not None and best.yaw_err is not None:
                    passed += 1 if best.pos_err < 0.25 and best.yaw_err < 0.17 else 0
                    correct += 1 if best.pos_err < 0.25 and best.yaw_err < 0.17 else 0
            else:
                print(f"  ⚠ NO CANDIDATE PASSED both right_ok and back_ok")
                # Print failure analysis
                print(f"  Failure analysis:")
                for r in results:
                    print(f"    [{r.candidate.id}] right_ok={r.right.ok} back_ok={r.back.ok} "
                          f"R(cnt={r.right.count} med={r.right.med_residual:.3f} sp={r.right.span:.2f}) "
                          f"B(cnt={r.back.count} med={r.back.med_residual:.3f} sp={r.back.span:.2f})")

    # Write CSV + JSONL
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    with open(jsonl_path, 'w') as f:
        for row in all_rows:
            f.write(json.dumps(row) + '\n')

    if total == 0:
        print("\nNo pointcloud data collected — is the simulation running and /lidar/points publishing?")
    else:
        print(f"\n{'='*60}")
        print(f"  ACCURACY: {correct}/{total}  (pos_err<0.25m & yaw_err<10°)")
        print(f"  PASS RATE: {passed}/{total}")
        print(f"  Logs: {csv_path}  {jsonl_path}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
