#!/usr/bin/env python3
"""
fix_geometry_validator.py — FIX-point geometry-based candidate selector.

Problem: corner_localizer produces multiple pose candidates; some are
"swapped" (east/north wall identities exchanged). NE corner scoring alone
can validate both correct and swapped candidates.

Solution: For every candidate, transform 5-frame LiDAR pointcloud to the
known FIX base_link frame. In FIX frame, walls have KNOWN positions:
  - right wall  (NORTH wall): y ≈ -1.00m  (4.50 - 3.50)
  - back wall   (EAST wall):  x ≈ -1.24m  (6.25 - 5.01)

Candidates that put walls at correct FIX-frame positions are valid.
Score = how well the walls align + point count + span.

Usage:
  python3 fix_geometry_validator.py --seed 0 --attempts 3
  python3 fix_geometry_validator.py --seeds 0,7,13,31,42 --attempts 3
"""

import sys, os, time, math, json, csv, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

# ─── Constants ───
FIX_X = 5.01
FIX_Y = 3.50
FIX_YAW = math.pi  # facing west (180°)

RIGHT_WALL_Y = -(4.50 - 3.50)   # -1.00 (NORTH wall in FIX frame)
BACK_WALL_X = -(6.25 - 5.01)    # -1.24 (EAST wall in FIX frame)

RIGHT_STRIP_HALF = 0.12         # |y - RIGHT_WALL_Y| < 0.12
BACK_STRIP_HALF = 0.12          # |x - BACK_WALL_X| < 0.12

MIN_RIGHT_COUNT = 800
MIN_BACK_COUNT = 500
MAX_MED_RES = 0.06
MIN_RIGHT_SPAN = 0.6
MIN_BACK_SPAN = 0.5

N_FRAMES = 5
VOXEL_SIZE = 0.03

# ─── Data structures ───

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

    def to_dict(self):
        return {"id": self.id, "x": self.x, "y": self.y,
                "yaw_deg": self.yaw_deg, "source": self.source,
                "orig_score": self.orig_score}

@dataclass
class StripMetrics:
    """Wall strip evaluation."""
    count: int = 0
    med_residual: float = 999.0
    p90_residual: float = 999.0
    span: float = 0.0
    density_bins: int = 0
    ok: bool = False

    def to_dict(self):
        return {"count": self.count, "med_residual": round(self.med_residual, 4),
                "p90_residual": round(self.p90_residual, 4),
                "span": round(self.span, 3), "density_bins": self.density_bins,
                "ok": self.ok}

@dataclass
class CandidateResult:
    candid: Candidate
    right: StripMetrics = field(default_factory=StripMetrics)
    back: StripMetrics = field(default_factory=StripMetrics)
    fix_score: float = 0.0
    selected: bool = False
    # GT (evaluation only, NOT for selection)
    gt_x: Optional[float] = None
    gt_y: Optional[float] = None
    gt_yaw: Optional[float] = None

    @property
    def pos_err(self):
        if self.gt_x is None:
            return None
        return math.hypot(self.candid.x - self.gt_x, self.candid.y - self.gt_y)

    @property
    def yaw_err(self):
        if self.gt_yaw is None:
            return None
        d = (self.candid.yaw - self.gt_yaw) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    def to_row(self, seed, attempt):
        return {
            "seed": seed, "attempt": attempt,
            "cand_id": self.candid.id,
            "source": self.candid.source,
            "x": self.candid.x, "y": self.candid.y,
            "yaw_deg": self.candid.yaw_deg,
            "orig_score": self.candid.orig_score,
            **{"right_" + k: v for k, v in self.right.to_dict().items()},
            **{"back_" + k: v for k, v in self.back.to_dict().items()},
            "fix_score": round(self.fix_score, 2),
            "selected": self.selected,
            "gt_x": self.gt_x, "gt_y": self.gt_y,
            "gt_yaw_deg": math.degrees(self.gt_yaw) if self.gt_yaw else None,
            "pos_err": round(self.pos_err, 4) if self.pos_err else None,
            "yaw_err_deg": round(math.degrees(self.yaw_err), 4) if self.yaw_err else None,
        }

# ─── Voxel grid (deterministic) ───

def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    """Deterministic voxel centroid downsampling."""
    if len(points) == 0:
        return points
    voxel_indices = np.floor(points[:, :3] / size).astype(np.int32)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return points[unique_idx]

# ─── Transform helpers ───

def rot_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])

def transform_points(points: np.ndarray, dx: float, dy: float, yaw: float) -> np.ndarray:
    """Rotate + translate 2D points (Nx3 array, uses columns 0,1)."""
    if len(points) == 0:
        return points
    R = rot_matrix(yaw)
    xy = points[:, :2] @ R.T
    out = points.copy()
    out[:, 0] = xy[:, 0] + dx
    out[:, 1] = xy[:, 1] + dy
    return out

def candidate_to_fix(points_base: np.ndarray, cand: Candidate) -> np.ndarray:
    """Transform points from candidate's base_link to FIX base_link.
    base_link → world → FIX_base_link
    """
    # base_link → world
    p_world = transform_points(points_base, cand.x, cand.y, cand.yaw)
    # world → FIX base_link
    p_fix = transform_points(p_world, -FIX_X, -FIX_Y, 0)
    p_fix = transform_points(p_fix, 0, 0, -FIX_YAW)
    return p_fix

# ─── Strip metrics ───

def strip_metrics(points_fix: np.ndarray, target_line: float, along_axis: int) -> StripMetrics:
    """Compute wall strip metrics.
    For right_wall: target_line = RIGHT_WALL_Y, along_axis = 0 (check y, span along x)
    For back_wall:  target_line = BACK_WALL_X,  along_axis = 1 (check x, span along y)
    """
    r = StripMetrics()

    normal_axis = 1 - along_axis  # 0→1, 1→0
    half = RIGHT_STRIP_HALF if normal_axis == 0 else BACK_STRIP_HALF  # both 0.12

    # Points in strip band
    residuals = np.abs(points_fix[:, normal_axis] - target_line)
    in_strip = residuals < half
    r.count = int(np.sum(in_strip))

    if r.count < 5:
        return r

    strip_pts = points_fix[in_strip]
    strip_res = residuals[in_strip]
    r.med_residual = float(np.median(strip_res))
    r.p90_residual = float(np.percentile(strip_res, 90))
    r.span = float(np.max(strip_pts[:, along_axis]) - np.min(strip_pts[:, along_axis]))

    # Density: count bins with >10 points along span
    bin_size = 0.1
    if r.span > 0:
        n_bins = max(1, int(r.span / bin_size))
        bins = np.floor((strip_pts[:, along_axis] - np.min(strip_pts[:, along_axis])) / bin_size).astype(int)
        bin_counts = np.bincount(np.clip(bins, 0, n_bins - 1), minlength=n_bins)
        r.density_bins = int(np.sum(bin_counts >= 10))

    # Hard checks
    r.ok = (r.count >= MIN_RIGHT_COUNT and r.med_residual <= MAX_MED_RES
            and r.span >= MIN_RIGHT_SPAN) if normal_axis == 0 else \
           (r.count >= MIN_BACK_COUNT and r.med_residual <= MAX_MED_RES
            and r.span >= MIN_BACK_SPAN)
    return r

# ─── Score calculation ───

def compute_fix_score(right: StripMetrics, back: StripMetrics) -> float:
    score = 0.0
    score += 80 if right.ok else -80
    score += 80 if back.ok else -80
    score += min(right.count, 3000) * 0.01
    score += min(back.count, 2500) * 0.01
    score += min(right.span, 2.0) * 10
    score += min(back.span, 2.0) * 10
    score -= 300 * right.med_residual
    score -= 300 * back.med_residual
    score -= 100 * max(0, right.p90_residual - 0.12)
    score -= 100 * max(0, back.p90_residual - 0.12)
    return score

# ─── Main validation ───

def validate_candidates(
    points_frames: List[np.ndarray],
    candidates: List[Candidate],
    gt: Optional[Tuple[float, float, float]] = None
) -> List[CandidateResult]:
    """Run FIX geometry validation on all candidates."""
    # Merge & voxel all frames
    merged = np.vstack(points_frames) if points_frames else np.zeros((0, 3))
    merged = voxel_downsample(merged, VOXEL_SIZE)

    results = []
    for c in candidates:
        p_fix = candidate_to_fix(merged, c)
        right = strip_metrics(p_fix, RIGHT_WALL_Y, along_axis=0)   # check y, span x
        back = strip_metrics(p_fix, BACK_WALL_X, along_axis=1)     # check x, span y
        score = compute_fix_score(right, back)
        r = CandidateResult(candid=c, right=right, back=back, fix_score=score)
        if gt:
            r.gt_x, r.gt_y, r.gt_yaw = gt
        results.append(r)

    # Selection: only candidates with right_ok AND back_ok
    valid = [r for r in results if r.right.ok and r.back.ok]
    if valid:
        best = max(valid, key=lambda r: r.fix_score)
        best.selected = True

    return results

# ─── Test runner ───

def run_test(seed: int, attempt: int) -> List[CandidateResult]:
    """Run one test: collect frames, get candidates, validate."""
    import rospy
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    import tf2_ros

    # Collect N_FRAMES pointclouds in base_link
    frames = []
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)

    for _ in range(N_FRAMES):
        try:
            msg = rospy.wait_for_message('/lidar/points', PointCloud2, timeout=2)
            # Transform to base_link
            try:
                transform = tf_buffer.lookup_transform('base_link', msg.header.frame_id,
                                                       msg.header.stamp, rospy.Duration(1.0))
                # For now, assume points already in base_link (lidar_frame=base_link)
            except:
                pass
            gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            pts = np.array(list(gen))
            # Filter range
            r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            pts = pts[(r > 0.35) & (r < 20.0)]
            frames.append(pts)
            rospy.sleep(0.1)
        except:
            continue

    if len(frames) < 2:
        print(f"  seed={seed} attempt={attempt}: only {len(frames)} frames, skip")
        return []

    # Mock candidates for standalone test (replace with actual corner_localizer)
    candidates = _get_mock_candidates(seed, attempt)

    gt = (FIX_X + np.random.normal(0, 0.1),
          FIX_Y + np.random.normal(0, 0.1),
          FIX_YAW + np.random.normal(0, 0.1))  # mock GT

    return validate_candidates(frames, candidates, gt)

def _get_mock_candidates(seed: int, attempt: int) -> List[Candidate]:
    """Mock candidates for standalone test. Replace with corner_localizer call."""
    np.random.seed(seed * 100 + attempt)
    cands = []
    # Correct candidate
    cands.append(Candidate(0, FIX_X + np.random.normal(0, 0.05),
                           FIX_Y + np.random.normal(0, 0.05),
                           FIX_YAW + np.random.normal(0, 0.05), source="corner", orig_score=0.9))
    # Swapped candidate
    cands.append(Candidate(1, FIX_X + np.random.normal(0, 0.05),
                           FIX_Y + np.random.normal(0, 0.05),
                           FIX_YAW + math.pi / 2 + np.random.normal(0, 0.1),
                           source="corner", orig_score=0.85))
    # Random noise candidates
    for i in range(2, 5):
        cands.append(Candidate(i, FIX_X + np.random.normal(0, 0.5),
                               FIX_Y + np.random.normal(0, 0.5),
                               np.random.uniform(0, 2 * math.pi),
                               source="corner", orig_score=0.5))
    return cands

# ─── CLI ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--attempts', type=int, default=3)
    args = parser.parse_args()

    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]

    csv_path = '/tmp/nav_test/logs/fix_validator.csv'
    jsonl_path = '/tmp/nav_test/logs/fix_validator.jsonl'
    os.makedirs('/tmp/nav_test/logs', exist_ok=True)

    fieldnames = ['seed', 'attempt', 'cand_id', 'source', 'x', 'y', 'yaw_deg',
                  'orig_score', 'right_count', 'right_med', 'right_p90', 'right_span',
                  'right_density_bins', 'right_ok', 'back_count', 'back_med', 'back_p90',
                  'back_span', 'back_density_bins', 'back_ok', 'fix_score', 'selected',
                  'gt_x', 'gt_y', 'gt_yaw_deg', 'pos_err', 'yaw_err_deg']

    all_rows = []
    correct_count = 0
    total = 0

    for seed in seeds:
        for a in range(args.attempts):
            results = run_test(seed, a)
            if not results:
                continue
            total += 1
            selected = [r for r in results if r.selected]
            print(f"\nSeed={seed} attempt={a}: {len(results)} candidates, "
                  f"selected={'YES' if selected else 'NONE'}")

            for r in results:
                row = r.to_row(seed, a)
                all_rows.append(row)

                marker = " ★" if r.selected else ""
                ok = "OK" if r.pos_err and r.pos_err < 0.25 and r.yaw_err and r.yaw_err < 0.17 else ""
                print(f"  cand={r.candid.id} score={r.fix_score:.1f} "
                      f"right(ok={r.right.ok} cnt={r.right.count} med={r.right.med_residual:.3f}) "
                      f"back(ok={r.back.ok} cnt={r.back.count} med={r.back.med_residual:.3f})"
                      f"  pos_err={r.pos_err:.3f} yaw_err={math.degrees(r.yaw_err) if r.yaw_err else math.nan:.1f}°{marker}{' '+ok}")

            if selected:
                best = selected[0]
                if best.pos_err is not None and best.yaw_err is not None:
                    if best.pos_err < 0.25 and best.yaw_err < 0.17:
                        correct_count += 1

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)

    # Write JSONL
    with open(jsonl_path, 'w') as f:
        for row in all_rows:
            f.write(json.dumps(row) + '\n')

    print(f"\n=== SUMMARY: {correct_count}/{total} correct (pos_err<0.25m & yaw_err<10°) ===")

if __name__ == '__main__':
    main()
