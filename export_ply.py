#!/usr/bin/env python3
"""Export R<12m PLY for all 5 seeds."""
import rospy, sys, os, time, math, numpy as np
os.environ["DISPLAY"] = ":0"
sys.path.insert(0, "/tmp/nav_test")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/utils")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher
import wall_corner_localizer as wcl

SEED = int(sys.argv[1])
OUTDIR = "/tmp/nav_test/logs/ply_export"
os.makedirs(OUTDIR, exist_ok=True)

launcher = SimLauncher(scene="scene1", seed=SEED, robot_version=52)
launcher.start(node_name=f"ply_s{SEED}", timeout=120)
time.sleep(8)

pts = wcl.collect_cloud(5)
if pts is None or len(pts) < 100:
    print("NO PTS"); launcher.stop(); exit()

r12 = pts[np.hypot(pts[:,0], pts[:,1]) < 12.0]
path = f"{OUTDIR}/seed{SEED}_r12.ply"
with open(path, "w") as f:
    f.write(f"ply\nformat ascii 1.0\ncomment seed={SEED} r<12m\n")
    f.write(f"element vertex {len(r12)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property float range\nend_header\n")
    for p in r12:
        r = math.hypot(p[0], p[1])
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f}\n")
print(f"WROTE {path} ({len(r12)} pts)")
launcher.stop()
