#!/usr/bin/env python3
"""verify_topedge_lines.py — Run seed 0, export debug PNGs + line diagnostics."""
import rospy, sys, os, time, math, numpy as np
os.environ["DISPLAY"] = ":0"
sys.path.insert(0,"/tmp/nav_test")
sys.path.insert(0,"/root/kuavo_ws/src/craic_simulator/utils")
sys.path.insert(0,"/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher
import corner_shape_localizer as csl

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ODIR = f"/tmp/nav_test/logs/topedge_s{SEED}"
os.makedirs(ODIR, exist_ok=True)

print(f"TOPDGV S{SEED} Starting SimLauncher...", flush=True)
launcher = SimLauncher(scene="scene1", seed=SEED, robot_version=52)
launcher.start(node_name=f"tdgv_s{SEED}", timeout=120)
print("SIM_READY", flush=True)
time.sleep(8)

pts = csl.collect_base_link_cloud(5)
if pts is None: print("NO PTS"); launcher.stop(); exit()
print(f"pts={len(pts)}", flush=True)

lines, pairs, wall_pts = csl.detect_topedge_lines(pts)
print(f"lines={len(lines)} pairs={len(pairs)} wall_pts={len(wall_pts)}", flush=True)

for i, ln in enumerate(lines):
    print(f"  L{i}: a={math.degrees(ln['alpha']):.0f}° rho={ln['rho']:.3f} inl={ln['inliers']} sp={ln['span']:.2f} r_med={ln['r_med']:.2f}", flush=True)

for i, (la, lb) in enumerate(pairs[:10]):
    da = math.degrees(abs(csl.angle_diff(la["alpha"], lb["alpha"])))
    print(f"  P{i}: L{lines.index(la)}-L{lines.index(lb)} angle={da:.1f}°", flush=True)

# ---------- PNGs ----------
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    r12 = pts[np.hypot(pts[:,0], pts[:,1]) < 12.0]

    # 1. Full point cloud R<12m
    fig,ax=plt.subplots(figsize=(10,10))
    ax.scatter(r12[:,0],r12[:,1],c='#cccccc',s=1,alpha=0.3)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Seed {SEED} — Full cloud R<12m"); ax.grid(True,alpha=0.2)
    plt.savefig(f"{ODIR}/full_cloud_r12.png",dpi=150); plt.close()
    print("full_cloud_r12.png", flush=True)

    # 2. wall_pts_debug.png — only top-edge wall points
    if len(wall_pts)>0:
        wr=np.hypot(wall_pts[:,0],wall_pts[:,1])
        fig,ax=plt.subplots(figsize=(10,10))
        sc=ax.scatter(wall_pts[:,0],wall_pts[:,1],c=wall_pts[:,2],s=3,cmap='viridis',alpha=0.7)
        plt.colorbar(sc,label='z')
        ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"Seed {SEED} — Top-edge wall points ({len(wall_pts)} pts)"); ax.grid(True,alpha=0.2)
        plt.savefig(f"{ODIR}/wall_pts_debug.png",dpi=150); plt.close()
        print("wall_pts_debug.png", flush=True)

    # 3. hough_lines_v2.png — refined lines on top-edge points
    if len(lines)>0:
        fig,ax=plt.subplots(figsize=(12,12))
        if len(wall_pts)>0:
            ax.scatter(wall_pts[:,0],wall_pts[:,1],c='#cccccc',s=2,alpha=0.4)
        colors=plt.cm.tab20(np.linspace(0,1,len(lines)))
        for i,ln in enumerate(lines):
            pts_inl=ln["points"]
            c,s=math.cos(ln["alpha"]),math.sin(ln["alpha"])
            t=np.array([-s,c]); v=np.dot(pts_inl,t)
            q05=float(np.percentile(v,5)); q95=float(np.percentile(v,95))
            p0=np.array([c*ln["rho"]-s*q05,s*ln["rho"]+c*q05])
            p1=np.array([c*ln["rho"]-s*q95,s*ln["rho"]+c*q95])
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color=colors[i],lw=2)
            ax.scatter(pts_inl[:,0],pts_inl[:,1],color=colors[i],s=3,alpha=0.3)
        ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"Seed {SEED} — {len(lines)} top-edge lines, {len(pairs)} orth pairs")
        ax.grid(True,alpha=0.2)
        plt.savefig(f"{ODIR}/hough_lines_v2.png",dpi=150); plt.close()
        print("hough_lines_v2.png", flush=True)

    print(f"\nDone → {ODIR}")
except Exception as e:
    print(f"PNG error: {e}")

launcher.stop()
