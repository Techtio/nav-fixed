#!/usr/bin/env python3
"""
Offline wall-corner localizer for CRAIC scene1.
Input: base_link point cloud CSV with columns x,y,z (or npz with arr_0 / points).
Output: Hough wall peaks and pose candidates (x,y,yaw) solved against known East/North walls.

This is intentionally simple:
  1) use a broad point filter, not fragile segment splitting;
  2) run multi-rho Hough so near false lines do not delete far true wall lines;
  3) enumerate both wall identities and signed rho variants;
  4) select by world-wall residual, not by local line votes only.
"""
import argparse, math, os
import numpy as np

EAST_X = 6.25
NORTH_Y = 4.50
SPAWN_X = (3.8, 5.8)
SPAWN_Y = (2.0, 3.95)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        return df[["x", "y", "z"]].to_numpy(dtype=float)
    if ext == ".npz":
        data = np.load(path)
        if "points" in data:
            return data["points"].astype(float)
        return data[list(data.keys())[0]].astype(float)
    raise ValueError("Use CSV or NPZ. For PLY, export x,y,z CSV first or add open3d loader.")


def filter_wall_points(points: np.ndarray, min_r=0.35, max_r=12.0,
                       z_min=-1.0, z_max=0.1) -> np.ndarray:
    """Broad filter. Do not over-filter: keep enough top-edge/wall points."""
    r = np.hypot(points[:, 0], points[:, 1])
    m = (r > min_r) & (r < max_r) & (points[:, 2] > z_min) & (points[:, 2] < z_max)
    return points[m]


def hough_peaks(points: np.ndarray, ang_step=1.0, rho_step=0.02, top=40):
    xy = points[:, :2]
    alphas = np.deg2rad(np.arange(-180, 180, ang_step))
    maxrho = float(np.max(np.linalg.norm(xy, axis=1)) + 0.5)
    rbins = np.arange(-maxrho, maxrho + rho_step, rho_step)

    acc = []
    for a in alphas:
        rho = xy[:, 0] * math.cos(a) + xy[:, 1] * math.sin(a)
        hist, _ = np.histogram(rho, bins=rbins)
        acc.append(hist)
    acc = np.asarray(acc)

    flat = acc.ravel()
    k = min(len(flat), top * 50)
    order = np.argpartition(flat, -k)[-k:]
    order = order[np.argsort(flat[order])[::-1]]

    peaks = []
    for idx in order:
        ai, ri = np.unravel_index(idx, acc.shape)
        a = float(alphas[ai])
        rho = float((rbins[ri] + rbins[ri + 1]) / 2)
        votes = int(acc[ai, ri])

        # Multi-rho dedupe: only merge nearly identical signed lines.
        # Do NOT merge all same-angle lines, otherwise a close tray/inner line can erase the real wall.
        duplicate = False
        for pa, pr, _ in peaks:
            da = abs((math.degrees(a - pa) + 90) % 180 - 90)
            if da < 2.5 and abs(rho - pr) < 0.08:
                duplicate = True
                break
        if not duplicate:
            peaks.append((a, rho, votes))
        if len(peaks) >= top:
            break
    return peaks


def line_variants(line):
    """Same physical line can be represented by (alpha,rho) or (alpha+pi,-rho)."""
    a, r, _ = line
    return [(a, r), (wrap(a + math.pi), -r)]


def solve_assignment(east_line, north_line):
    """Given line identities, solve robot pose against world East/North walls."""
    ae, re = east_line
    an, rn = north_line

    # In base_link: world east-wall normal is rotated by -yaw.
    yaw_e = wrap(-ae)
    # World north-wall normal points +Y, angle pi/2 in world.
    yaw_n = wrap(math.pi / 2 - an)
    ycon = abs(math.degrees(wrap(yaw_e - yaw_n)))
    if ycon > 8.0:
        return None

    yaw = math.atan2(math.sin(yaw_e) + math.sin(yaw_n),
                     math.cos(yaw_e) + math.cos(yaw_n))
    x = EAST_X - re
    y = NORTH_Y - rn
    if not (SPAWN_X[0] <= x <= SPAWN_X[1] and SPAWN_Y[0] <= y <= SPAWN_Y[1]):
        return None
    return {"x": x, "y": y, "yaw": wrap(yaw), "yaw_consistency_deg": ycon}


def world_wall_residual(points: np.ndarray, pose, band=0.08):
    """Transform base_link points with candidate pose, then check support on known walls."""
    x, y, yaw = pose["x"], pose["y"], pose["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    X = c * points[:, 0] - s * points[:, 1] + x
    Y = s * points[:, 0] + c * points[:, 1] + y

    de = np.abs(X - EAST_X)
    dn = np.abs(Y - NORTH_Y)
    me = de < band
    mn = dn < band
    east_count = int(me.sum())
    north_count = int(mn.sum())
    east_med = float(np.median(de[me])) if east_count > 20 else 9.0
    north_med = float(np.median(dn[mn])) if north_count > 20 else 9.0
    return east_count, east_med, north_count, north_med


def detect_corner_pose(points: np.ndarray, verbose=False):
    P = filter_wall_points(points)
    peaks = hough_peaks(P, top=40)
    candidates = []

    for i, la in enumerate(peaks):
        for j, lb in enumerate(peaks):
            if i >= j:
                continue
            da = abs((math.degrees(la[0] - lb[0]) + 90) % 180 - 90)
            if abs(da - 90) > 6:
                continue

            # Try both identities: A=east/B=north and B=east/A=north.
            for ae, re in line_variants(la):
                for an, rn in line_variants(lb):
                    sol = solve_assignment((ae, re), (an, rn))
                    if sol is not None:
                        ec, em, nc, nm = world_wall_residual(P, sol)
                        score = -(em + nm) + 0.0002 * (ec + nc) - 0.01 * sol["yaw_consistency_deg"]
                        candidates.append({**sol, "score": score, "identity": "A_east_B_north",
                                           "line_i": i, "line_j": j,
                                           "east_count": ec, "east_med": em,
                                           "north_count": nc, "north_med": nm})
                    sol = solve_assignment((an, rn), (ae, re))
                    if sol is not None:
                        ec, em, nc, nm = world_wall_residual(P, sol)
                        score = -(em + nm) + 0.0002 * (ec + nc) - 0.01 * sol["yaw_consistency_deg"]
                        candidates.append({**sol, "score": score, "identity": "B_east_A_north",
                                           "line_i": i, "line_j": j,
                                           "east_count": ec, "east_med": em,
                                           "north_count": nc, "north_med": nm})

    candidates.sort(key=lambda d: d["score"], reverse=True)
    if verbose:
        print(f"filtered_points={len(P)} peaks={len(peaks)} candidates={len(candidates)}")
        print("Top Hough peaks:")
        for k, (a, rho, votes) in enumerate(peaks[:12]):
            print(f"  L{k:02d}: alpha={math.degrees(a):7.1f} rho={rho:7.3f} votes={votes}")
        print("Top candidates:")
        for c in candidates[:20]:
            print(f"  score={c['score']:7.3f} {c['identity']} L{c['line_i']},L{c['line_j']} "
                  f"pose=({c['x']:.3f},{c['y']:.3f},{math.degrees(c['yaw']):.1f}) "
                  f"ycon={c['yaw_consistency_deg']:.1f} "
                  f"east={c['east_count']}/{c['east_med']:.3f} "
                  f"north={c['north_count']}/{c['north_med']:.3f}")
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud", nargs="?", default="/mnt/data/all_points.csv")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    pts = load_points(args.cloud)
    candidates = detect_corner_pose(pts, verbose=True)
    if not candidates:
        print("NO_CANDIDATES")
        return
    best = candidates[0]
    print("\nBEST:",
          f"x={best['x']:.3f}", f"y={best['y']:.3f}",
          f"yaw_deg={math.degrees(best['yaw']):.1f}", f"score={best['score']:.3f}")


if __name__ == "__main__":
    main()
