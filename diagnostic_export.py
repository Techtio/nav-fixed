#!/usr/bin/env python3
"""
diagnostic_export.py — Export full point cloud diagnostics for one seed/attempt.
Run: python3 diagnostic_export.py <seed> [attempt=1]
"""
import rospy, sys, os, time, math, numpy as np

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ATT = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ODIR = f"/tmp/nav_test/logs/diag_s{SEED}_a{ATT}"
os.makedirs(ODIR, exist_ok=True)

os.environ["DISPLAY"] = ":0"
sys.path.insert(0, "/tmp/nav_test")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/utils")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher
import corner_shape_localizer as csl

print(f"DIAG seed={SEED} att={ATT} Starting SimLauncher...", flush=True)
launcher = SimLauncher(scene="scene1", seed=SEED, robot_version=52)
launcher.start(node_name=f"diag_s{SEED}", timeout=120)
print("SIM_READY", flush=True)
time.sleep(8)

# 1. Collect raw clouds
pts_base = csl.collect_base_link_cloud(5)
if pts_base is None or len(pts_base) < 100:
    print("NO POINTS"); launcher.stop(); exit()

# Save raw + r12 versions
def save_ply(pts, path, comment=""):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment {}\n".format(comment))
        f.write("element vertex {}\n".format(len(pts)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float range\nproperty float angle\nend_header\n")
        for p in pts:
            r = math.hypot(p[0], p[1]); a = math.degrees(math.atan2(p[1], p[0]))
            f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(p[0], p[1], p[2], r, a))
    print(f"  wrote {path} ({len(pts)} pts)")

save_ply(pts_base, f"{ODIR}/raw_base_link.ply", f"seed={SEED} attempt={ATT}")

r12 = pts_base[np.hypot(pts_base[:,0], pts_base[:,1]) < 12.0]
save_ply(r12, f"{ODIR}/base_link_r12.ply", f"seed={SEED} r<12m")

# Wall candidate points
wall = csl.extract_wall_points(pts_base)
if len(wall) > 0:
    save_ply(wall, f"{ODIR}/wall_candidate_points.ply", f"seed={SEED} wall extract")

# 2. CSV: all points
pts_csv = np.column_stack([pts_base[:,:3],
    np.hypot(pts_base[:,0], pts_base[:,1]),
    np.arctan2(pts_base[:,1], pts_base[:,0])])
np.savetxt(f"{ODIR}/all_points.csv", pts_csv, delimiter=",",
           header="x,y,z,r,angle", fmt="%.6f", comments="")

# 3. Run shape localizer
segs, n_raw = csl.detect_wall_segments(wall[:,:2])
lines_data = []
for s in segs:
    pts = s.points
    z_vals = wall[np.isin(np.arange(len(wall)), 
        np.where(np.isin(wall[:,:2], pts).all(axis=1))[0])[:1]] if len(wall) >= max(len(pts), 1) else pts
    r_vals = np.hypot(pts[:,0], pts[:,1])
    lines_data.append({
        "line_id": s.id, "alpha_deg": round(math.degrees(s.alpha), 2),
        "rho": round(s.rho, 4), "inliers": s.inliers,
        "span": round(s.span, 3), "votes": s.votes,
        "z_min": None, "z_max": None,
        "r_min": round(float(np.min(r_vals)), 3),
        "r_max": round(float(np.max(r_vals)), 3),
    })

import csv
with open(f"{ODIR}/detected_lines.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["line_id","alpha_deg","rho","inliers","span","votes","z_min","z_max","r_min","r_max"])
    w.writeheader(); w.writerows(lines_data)
print(f"  {len(lines_data)} lines exported")

# 4. Run full pipeline for candidate list
result, diag = csl.corner_shape_localize()
gt = csl.GT_POSES.get(SEED)
cand_rows = []
if result:
    # Add selected
    pose = result["pose"]
    pe = math.hypot(pose[0]-gt[0], pose[1]-gt[1]) if gt else None
    ye = min(abs(csl.wrap(pose[2]-gt[2])), 2*math.pi-abs(csl.wrap(pose[2]-gt[2]))) if gt else None
    cand_rows.append({
        "candidate_id": 0, "x": round(pose[0], 4), "y": round(pose[1], 4),
        "yaw_deg": round(math.degrees(pose[2]), 2),
        "line_east_id": result["la"].id, "line_north_id": result["lb"].id,
        "shape_score": round(result["shape_score"], 2),
        "fix_score": round(result["fix_score"], 2),
        "total_score": round(result["total_score"], 2),
        "selected": True,
        "pos_err": round(pe, 4) if pe else None,
        "yaw_err_deg": round(math.degrees(ye), 1) if ye else None,
    })
    # Other candidates
    for i, c in enumerate(result.get("all_candidates", [])[:20]):
        if any(k in c for k in ["pose"]):
            p = c["pose"]
            pe2 = math.hypot(p[0]-gt[0], p[1]-gt[1]) if gt else None
            ye2 = min(abs(csl.wrap(p[2]-gt[2])), 2*math.pi-abs(csl.wrap(p[2]-gt[2]))) if gt else None
            cand_rows.append({
                "candidate_id": i+1, "x": round(p[0], 4), "y": round(p[1], 4),
                "yaw_deg": round(math.degrees(p[2]), 2),
                "line_east_id": getattr(c.get("la",{}),"id",-1),
                "line_north_id": getattr(c.get("lb",{}),"id",-1),
                "shape_score": round(c.get("shape_score", 0), 2),
                "fix_score": round(c.get("fix_score", 0), 2),
                "total_score": round(c.get("total_score", 0), 2),
                "selected": False,
                "pos_err": round(pe2, 4) if pe2 else None,
                "yaw_err_deg": round(math.degrees(ye2), 1) if ye2 else None,
            })

with open(f"{ODIR}/candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["candidate_id","x","y","yaw_deg",
        "line_east_id","line_north_id","shape_score","fix_score","total_score",
        "selected","pos_err","yaw_err_deg"])
    w.writeheader(); w.writerows(cand_rows)
print(f"  {len(cand_rows)} candidates exported")

# 5. PNG plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # XY by z
    fig, ax = plt.subplots(figsize=(10,10))
    sc = ax.scatter(pts_base[:,0], pts_base[:,1], c=pts_base[:,2], s=1, cmap='viridis', alpha=0.5)
    plt.colorbar(sc, label='z (m)')
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel("x base_link (forward)"); ax.set_ylabel("y base_link (left)")
    ax.set_title(f"Seed {SEED} Attempt {ATT} — XY by Z")
    ax.axhline(y=0, color='gray', ls='--', alpha=0.3); ax.axvline(x=0, color='gray', ls='--', alpha=0.3)
    plt.savefig(f"{ODIR}/xy_by_z.png", dpi=150); plt.close()
    print("  xy_by_z.png saved")

    # XY by range
    fig, ax = plt.subplots(figsize=(10,10))
    ranges = np.hypot(pts_base[:,0], pts_base[:,1])
    sc = ax.scatter(pts_base[:,0], pts_base[:,1], c=ranges, s=1, cmap='plasma', alpha=0.5)
    plt.colorbar(sc, label='range (m)')
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Seed {SEED} Attempt {ATT} — XY by Range")
    plt.savefig(f"{ODIR}/xy_by_range.png", dpi=150); plt.close()
    print("  xy_by_range.png saved")

    # Hough lines overlay
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(pts_base[:,0], pts_base[:,1], c='#cccccc', s=1, alpha=0.3)
    colors = plt.cm.tab20(np.linspace(0, 1, len(segs)))
    for i, s in enumerate(segs):
        p_start = s.p_start; p_end = s.p_end
        ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]],
                color=colors[i], lw=2, label=f"L{i}" if i < 20 else "")
        ax.scatter(s.points[:,0], s.points[:,1], color=colors[i], s=2, alpha=0.5)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Seed {SEED} — {len(segs)} detected segments")
    plt.savefig(f"{ODIR}/hough_lines.png", dpi=150); plt.close()
    print("  hough_lines.png saved")

    print(f"\nDone → {ODIR}")
except Exception as e:
    print(f"  PNG error: {e}")

launcher.stop()
