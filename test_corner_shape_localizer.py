#!/usr/bin/env python3
"""test_corner_shape_localizer v2 — 5 seeds × 3 attempts with pipeline diagnostics."""
import rospy, sys, os, time, math, json, csv
os.environ["DISPLAY"] = ":0"

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = f"/tmp/nav_test/logs/shape_v2_s{SEED}.json"

sys.path.insert(0, "/tmp/nav_test")
import corner_shape_localizer as csl

sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/utils")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher

print(f"SHAPE S{SEED} Starting SimLauncher...", flush=True)
launcher = SimLauncher(scene="scene1", seed=SEED, robot_version=52)
launcher.start(node_name=f"shp2_s{SEED}", timeout=120)
print(f"SHAPE S{SEED} SIM_READY", flush=True)
time.sleep(8)

gt = csl.GT_POSES.get(SEED)
print(f"GT: ({gt[0]:.3f},{gt[1]:.3f},{math.degrees(gt[2]):.0f}°)" if gt else "GT: N/A", flush=True)

COLS = ['seed','attempt','n_points','n_lines_raw','n_lines_kept','n_pairs',
        'n_pose_cands','n_after_fix','candidate_id','line_a_id','line_b_id',
        'x','y','yaw_deg','line_a_alpha','line_a_rho','line_a_span','line_a_votes',
        'line_b_alpha','line_b_rho','line_b_span','line_b_votes',
        'orth_err_deg','corner_gap','shape_score','wall_residual','ray_score',
        'right_count','right_med','right_p90','right_span','right_ok',
        'back_count','back_med','back_p90','back_span','back_ok',
        'total_score','selected',
        'gt_x','gt_y','gt_yaw_deg','pos_err','yaw_err_deg']
all_rows = []
summary = {}

for att in range(3):
    time.sleep(1)
    result, diag = csl.corner_shape_localize()
    print(f"  A{att+1} pts={diag.get('n_points','?')} lines_raw={diag.get('n_lines_raw','?')} "
          f"kept={diag.get('n_lines_kept','?')} pairs={diag.get('n_pairs','?')} "
          f"poses={diag.get('n_pose_cands','?')} fix={diag.get('n_after_fix','?')}", flush=True)

    if result is None:
        summary[f"A{att+1}"] = "NO_RESULT"
        all_rows.append({"seed":SEED,"attempt":att+1,"selected":False,
                         "n_points":diag.get('n_points',0),"n_lines_raw":diag.get('n_lines_raw',0),
                         "n_lines_kept":diag.get('n_lines_kept',0),"n_pairs":diag.get('n_pairs',0),
                         "n_pose_cands":diag.get('n_pose_cands',0),"n_after_fix":diag.get('n_after_fix',0)})
        if 'all_candidates' in diag:
            # Some candidates exist but none passed FIX
            for i, c in enumerate(diag.get('all_candidates', [])[:10]):
                fm = c.get('fix_metrics', {})
                print(f"    C{i}: fix_ok={c.get('fix_ok')} "
                      f"R(n={fm.get('right',{}).get('count',0)} m={fm.get('right',{}).get('med',999):.3f} ok={fm.get('right_ok')}) "
                      f"B(n={fm.get('back',{}).get('count',0)} m={fm.get('back',{}).get('med',999):.3f} ok={fm.get('back_ok')})")
        continue

    pose = result["pose"]; x, y, yaw = pose
    pe=None; ye=None
    if gt: pe=math.hypot(x-gt[0],y-gt[1]); ye=min(abs(csl.wrap(yaw-gt[2])), 2*math.pi-abs(csl.wrap(yaw-gt[2])))
    fm = result["fix_metrics"]
    correct = pe and ye and pe<0.25 and ye<0.175
    tag = "CORRECT" if correct else "WRONG"; summary[f"A{att+1}"] = tag

    row = {"seed":SEED,"attempt":att+1,"candidate_id":0,
           "n_points":diag["n_points"],"n_lines_raw":diag["n_lines_raw"],
           "n_lines_kept":diag["n_lines_kept"],"n_pairs":diag["n_pairs"],
           "n_pose_cands":diag["n_pose_cands"],"n_after_fix":diag["n_after_fix"],
           "line_a_id":result["la"].id,"line_b_id":result["lb"].id,
           "x":round(x,4),"y":round(y,4),"yaw_deg":round(math.degrees(yaw),2),
           "line_a_alpha":round(result["la"].alpha,4),"line_a_rho":round(result["la"].rho,4),
           "line_a_span":round(result["la"].span,3),"line_a_votes":result["la"].votes,
           "line_b_alpha":round(result["lb"].alpha,4),"line_b_rho":round(result["lb"].rho,4),
           "line_b_span":round(result["lb"].span,3),"line_b_votes":result["lb"].votes,
           "orth_err_deg":round(result["orth_err_deg"],2),
           "corner_gap":round(result["corner_gap"],3),
           "shape_score":round(result["shape_score"],2),
           "wall_residual":round(result["wall_residual"],3),
           "ray_score":round(result["ray_score"],2),
           "right_count":fm["right"]["count"],"right_med":round(fm["right"]["med"],4),
           "right_p90":round(fm["right"]["p90"],4),"right_span":round(fm["right"]["span"],3),
           "right_ok":fm["right_ok"],"back_count":fm["back"]["count"],
           "back_med":round(fm["back"]["med"],4),"back_p90":round(fm["back"]["p90"],4),
           "back_span":round(fm["back"]["span"],3),"back_ok":fm["back_ok"],
           "total_score":round(result["total_score"],2),"selected":True,
           "gt_x":round(gt[0],3) if gt else None,"gt_y":round(gt[1],3) if gt else None,
           "gt_yaw_deg":round(math.degrees(gt[2]),1) if gt else None,
           "pos_err":round(pe,4) if pe else None,"yaw_err_deg":round(math.degrees(ye),1) if ye else None}
    all_rows.append(row)

    print(f"  {tag} {'✓' if correct else '✗'} "
          f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}°) "
          f"pe={pe:.3f}m ye={math.degrees(ye):.1f}°" if pe else f"" if ye else f"",
          flush=True)
    if not correct:
        print(f"    line_a=[{result['la'].alpha:.2f}, rho={result['la'].rho:.2f}, span={result['la'].span:.2f}, votes={result['la'].votes}]", flush=True)
        print(f"    line_b=[{result['lb'].alpha:.2f}, rho={result['lb'].rho:.2f}, span={result['lb'].span:.2f}, votes={result['lb'].votes}]", flush=True)

    # Show other FIX-valid candidates
    top = sorted(result.get("all_candidates", []), key=lambda c: c.get("total_score", -1), reverse=True)
    print(f"    top FIX-ok: {len([c for c in top if c.get('fix_ok')])}/{len(top)}", flush=True)
    for j, c in enumerate(top[:3]):
        fm2 = c.get("fix_metrics", {})
        sc = c.get("total_score", -1)
        pe2 = math.hypot(c["pose"][0]-gt[0], c["pose"][1]-gt[1]) if gt else -1
        print(f"      #{j}: score={sc:.1f} pe={pe2:.3f}m fix_ok={c.get('fix_ok')} "
              f"R(n={fm2.get('right',{}).get('count',0)} m={fm2.get('right',{}).get('med',999):.3f}) "
              f"B(n={fm2.get('back',{}).get('count',0)} m={fm2.get('back',{}).get('med',999):.3f})", flush=True)

for att in range(len(all_rows), 3):
    all_rows.append({"seed":SEED,"attempt":att+1,"selected":False})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f: json.dump(all_rows, f)
print(f"SHAPE S{SEED} DONE {len(all_rows)} rows  summary={summary}", flush=True)
launcher.stop()
