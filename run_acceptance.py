#!/usr/bin/env python3
"""Sequential 15-test acceptance run for fixed_pose_validator.
5 seeds × 3 attempts. Host-side controller."""
import subprocess, time, sys, os

SEEDS = [0, 7, 13, 31, 42]
C = "kuavo_clean_craic"
ATTEMPTS = 3

def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except: return ""

for seed in SEEDS:
    print(f"\n{'='*50}\n  SEED {seed}\n{'='*50}")
    
    # Kill old sim
    run(f"docker exec {C} pkill -f ros", timeout=5)
    time.sleep(5)
    
    # Start simulation via SimLauncher script
    script = f'''
import rospy, sys, os, time
os.environ["DISPLAY"] = ":0"
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/utils")
sys.path.insert(0, "/root/kuavo_ws/src/craic_simulator/lib")
from sim_launcher import SimLauncher
launcher = SimLauncher(scene="scene1", seed={seed}, robot_version=52)
launcher.start(node_name="fv_s{seed}", timeout=120)
print("SIM_READY", flush=True)
time.sleep(5)

sys.path.insert(0, "/tmp/nav_test")
import corner_localizer as cl
for att in range({ATTEMPTS}):
    time.sleep(2)
    xy, xy3, ang, ran = cl.collect_base_link_cloud(5)
    if xy is None or len(xy) < 30:
        print(f"ATTEMPT{{att+1}} SKIP: no cloud")
        continue
    lines = cl.detect_lines(xy, xy3)
    if len(lines) < 2:
        print(f"ATTEMPT{{att+1}} SKIP: <2 lines")
        continue
    pairs, _ = cl.find_orthogonal_pairs(lines)
    if not pairs:
        print(f"ATTEMPT{{att+1}} SKIP: no pairs")
        continue
    allr = []
    for p in pairs:
        allr.extend(cl.generate_candidates_for_pair(p["li"], p["lj"], xy3, ang, ran))
    if not allr:
        print(f"ATTEMPT{{att+1}} SKIP: no candidates")
        continue
    allr.sort(key=lambda c: -c["score"])
    gt = cl.GT_POSES.get({seed})
    if gt: print(f"GT: x={{gt[0]:.2f}} y={{gt[1]:.2f}} yaw={{gt[2]*57.3:.0f}}°")
    for i, c in enumerate(allr[:10]):
        pe = ((c["x"]-gt[0])**2+(c["y"]-gt[1])**2)**0.5 if gt else -1
        ye = abs(c["yaw_deg"]-gt[2]*57.3) if gt else -1
        ok = "OK" if pe > 0 and pe < 0.25 and ye < 10 else ""
        print(f"A{{att+1}} C{{i}} x={{c[\"x\"]:.2f}} y={{c[\"y\"]:.2f}} yaw={{c[\"yaw_deg\"]:.1f}}° s={{c[\"score\"]:.1f}} err={{pe:.3f}}m/{{ye:.1f}}° {{ok}}")
print("DONE_" + str({seed}))
'''
    script_file = f"/tmp/fv_seed{seed}.py"
    with open(script_file, "w") as f:
        f.write(script)
    
    run(f"docker cp {script_file} {C}:/tmp/")
    out = run(f"docker exec -e DISPLAY=:0 {C} bash -c 'source /opt/ros/noetic/setup.bash && source /root/kuavo_ws/devel/setup.bash && export LD_LIBRARY_PATH=/opt/drake/lib:/root/kuavo_ws/installed/lib:$LD_LIBRARY_PATH && python3 -u {script_file}'", timeout=300)
    
    for line in out.split('\n'):
        if any(x in line for x in ['GT:', 'C0', 'C1', 'C2', 'C3', 'C4', 'DONE_', 'SKIP', 'sim_launcher']):
            print(line)
    
    print(f"SEED {seed} DONE")
    time.sleep(3)

print("\n=== ALL 15 TESTS COMPLETE ===")
