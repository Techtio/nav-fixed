#!/usr/bin/env python3
"""Test topedge_corner_localizer v3 — 5×3 with segment-based L-shapes."""
import rospy, sys, os, time, math, csv
os.environ["DISPLAY"]=":0"
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0
OUT=f"/tmp/nav_test/logs/tpe3_s{SEED}.csv"

sys.path.insert(0,"/tmp/nav_test"); sys.path.insert(0,"/root/kuavo_ws/src/craic_simulator/utils"); sys.path.insert(0,"/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher
import topedge_corner_localizer as tcl

print(f"TPE3 S{SEED} SimLauncher...",flush=True)
launcher=SimLauncher(scene="scene1",seed=SEED,robot_version=52)
launcher.start(node_name=f"tpe3_s{SEED}",timeout=120)
print("SIM_READY",flush=True); time.sleep(8)

gt=tcl.GT_POSES.get(SEED)
COLS=['seed','attempt','candidate_id','x','y','yaw_deg','pos_err','yaw_err_deg','selected',
      'stage','n_topedge','n_segments','n_lshapes','n_candidates',
      'full_score','sa_len','sb_len','sa_count','sb_count','sa_density','sb_density',
      'sa_alpha_deg','sb_alpha_deg','angle_diff_deg','corner_x','corner_y']
all_rows=[]

for att in range(3):
    time.sleep(1)
    pts=tcl.collect_base_link_cloud(5)
    cands,diag=tcl.localize_by_topedge_lshape(pts)
    st=diag.get("stage","?")

    print(f"  A{att+1} st={st} edge={diag.get('n_topedge',0)} segs={diag.get('n_segments',0)} "
          f"ls={diag.get('n_lshapes',0)} cands={diag.get('n_candidates',0)}",flush=True)

    if not cands:
        all_rows.append({"seed":SEED,"attempt":att+1,"selected":False,"stage":st,
                         "n_topedge":diag.get("n_topedge",0),"n_segments":diag.get("n_segments",0),
                         "n_lshapes":diag.get("n_lshapes",0),"n_candidates":0})
        continue

    for i,c in enumerate(cands):
        pose=c["pose"]; x,y,yaw=pose[0],pose[1],pose[2]; yc=pose[3] if len(pose)>3 else 0
        pe=math.hypot(x-gt[0],y-gt[1]) if gt else None
        ye=min(abs(tcl.wrap(yaw-gt[2])),2*math.pi-abs(tcl.wrap(yaw-gt[2]))) if gt else None
        sel=(i==0)
        correct=pe and ye and pe<.25 and ye<.175; mark="✓" if correct else "✗"
        sa=c["sa"]; sb=c["sb"]
        row={"seed":SEED,"attempt":att+1,"candidate_id":i,"x":round(x,4),"y":round(y,4),
             "yaw_deg":round(math.degrees(yaw),2),"pos_err":round(pe,4) if pe else None,
             "yaw_err_deg":round(math.degrees(ye),1) if ye else None,"selected":sel,
             "stage":st,"n_topedge":diag.get("n_topedge",0),"n_segments":diag.get("n_segments",0),
             "n_lshapes":diag.get("n_lshapes",0),"n_candidates":diag.get("n_candidates",0),
             "full_score":round(c["full_score"],1),"sa_len":round(sa["length"],3),"sb_len":round(sb["length"],3),
             "sa_count":sa["count"],"sb_count":sb["count"],
             "sa_density":round(sa["density"],1),"sb_density":round(sb["density"],1),
             "sa_alpha_deg":round(math.degrees(sa["alpha"]),1),"sb_alpha_deg":round(math.degrees(sb["alpha"]),1),
             "angle_diff_deg":round(abs(tcl.adiff_deg(sa["alpha"],sb["alpha"])),1),
             "corner_x":round(c["corner"][0],3),"corner_y":round(c["corner"][1],3)}
        all_rows.append(row)
        if sel or correct:
            print(f"    C{i} {mark} {'SELECTED' if sel else 'gt_ok'} "
                  f"pe={pe:.3f}m ye={math.degrees(ye):.1f}° "
                  f"sa_len={sa['length']:.2f} sb_len={sb['length']:.2f} "
                  f"ang={abs(tcl.adiff_deg(sa['alpha'],sb['alpha'])):.0f}° "
                  f"n={sa['count']}/{sb['count']}",flush=True)

os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=COLS,extrasaction='ignore'); w.writeheader(); w.writerows(all_rows)
print(f"TPE3 S{SEED} DONE {len(all_rows)} rows",flush=True)
launcher.stop()
