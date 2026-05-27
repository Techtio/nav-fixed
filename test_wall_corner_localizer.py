#!/usr/bin/env python3
"""Test wall_corner_localizer v3 — stabilized + per-frame QA + adaptive z."""
import rospy,sys,os,time,math,csv
os.environ["DISPLAY"]=":0"; SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0
OUT=f"/tmp/nav_test/logs/wcl3_s{SEED}.csv"; FOUT=f"/tmp/nav_test/logs/wcl3_frames_s{SEED}.csv"

sys.path.insert(0,"/tmp/nav_test"); sys.path.insert(0,"/root/kuavo_ws/src/craic_simulator/utils"); sys.path.insert(0,"/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher; import wall_corner_localizer as wcl

print(f"WCL3 S{SEED} SimLauncher...",flush=True)
launcher=SimLauncher(scene="scene1",seed=SEED,robot_version=52)
launcher.start(node_name=f"wcl3_s{SEED}",timeout=120)
print("SIM_READY",flush=True)
wcl.stabilize_robot()  # zero cmd_vel, wait 3s

gt=wcl.GT_POSES.get(SEED)
COLS=['seed','attempt','candidate_id','x','y','yaw_deg','pos_err','yaw_err_deg','selected',
      'score','identity','line_i','line_j','east_count','east_med','north_count','north_med','yaw_consistency_deg','n_peaks']
FCOLS=['seed','attempt','frame_id','n_points','z_p5','z_p50','z_p95','r_p50','r_p90','hough_raw','used']
rows=[]; frows=[]

for att in range(3):
    time.sleep(1)
    pts,frame_diags=wcl.collect_quality_frames(n_target=5,n_max=10)
    for d in frame_diags: d["seed"]=SEED; d["attempt"]=att+1; frows.append(d)

    n_used=sum(1 for d in frame_diags if d["used"]); n_tot=len(frame_diags)
    if pts is None:
        print(f"  A{att+1} NO_PTS",flush=True); rows.append({"seed":SEED,"attempt":att+1,"selected":False}); continue

    cands,peaks,diag=wcl.corner_localize(pts)
    pk=diag.get("n_peaks",0); nc=diag.get("n_cands",0)
    print(f"  A{att+1} pts={len(pts)} frames={n_used}/{n_tot} peaks={pk} cands={nc}",flush=True)

    if not cands:
        rows.append({"seed":SEED,"attempt":att+1,"selected":False,"n_peaks":pk}); continue

    for i,c in enumerate(cands):
        pe=math.hypot(c["x"]-gt[0],c["y"]-gt[1]) if gt else None
        ye=min(abs(wcl.wrap(c["yaw"]-gt[2])),2*math.pi-abs(wcl.wrap(c["yaw"]-gt[2]))) if gt else None
        sel=(i==0); correct=pe and ye and pe<.25 and ye<.175; mark="✓" if correct else "✗"
        rows.append({"seed":SEED,"attempt":att+1,"candidate_id":i,
            "x":round(c["x"],4),"y":round(c["y"],4),"yaw_deg":round(math.degrees(c["yaw"]),2),
            "pos_err":round(pe,4) if pe else None,"yaw_err_deg":round(math.degrees(ye),1) if ye else None,
            "selected":sel,"score":round(c["score"],4),"identity":c["identity"],
            "line_i":c["line_i"],"line_j":c["line_j"],
            "east_count":c["east_count"],"east_med":round(c["east_med"],4),
            "north_count":c["north_count"],"north_med":round(c["north_med"],4),
            "yaw_consistency_deg":round(c["yaw_consistency_deg"],1),"n_peaks":pk})
        if sel or correct:
            print(f"    C{i} {'SELECTED' if sel else 'gt_ok'} {mark} pe={pe:.3f}m ye={math.degrees(ye):.1f}° sc={c['score']:.3f} {c['identity']} L{c['line_i']},{c['line_j']}",flush=True)

os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,"w",newline="") as f: w=csv.DictWriter(f,fieldnames=COLS,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
with open(FOUT,"w",newline="") as f: w=csv.DictWriter(f,fieldnames=FCOLS,extrasaction='ignore'); w.writeheader(); w.writerows(frows)
print(f"WCL3 S{SEED} DONE {len(rows)} candidates + {len(frows)} frames",flush=True)
launcher.stop()
