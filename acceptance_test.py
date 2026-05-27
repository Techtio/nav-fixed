#!/usr/bin/env python3
"""Sequential acceptance test for fixed_pose_validator.
One seed at a time: restart sim → SimLauncher → validate → next seed."""
import subprocess, time, sys, os

CONTAINER = "kuavo_clean_craic"
SEEDS = [0, 7, 13, 31, 42]
ATTEMPTS = 3

def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except: return None

def docker(cmd, timeout=30):
    return run(f"docker exec {CONTAINER} bash -c 'source /opt/ros/noetic/setup.bash && source /root/kuavo_ws/devel/setup.bash && export LD_LIBRARY_PATH=/opt/drake/lib:/root/kuavo_ws/installed/lib:$LD_LIBRARY_PATH && {cmd}'", timeout)

for seed in SEEDS:
    print(f"\n{'='*50}\n  SEED {seed}\n{'='*50}")
    
    # Kill old sim
    run(f"docker exec {CONTAINER} pkill -f rosmaster", timeout=5)
    run(f"docker exec {CONTAINER} pkill -f roslaunch", timeout=5)
    time.sleep(5)
    
    # Run fixed_pose_validator for this seed
    out = docker(
        f"cd /tmp/nav_test && python3 -u fixed_pose_validator.py --seed {seed} --attempts {ATTEMPTS}",
        timeout=300
    )
    if out:
        for line in out.split('\n'):
            if '──' in line or '★' in line or '⚠' in line or 'ACCURACY' in line or 'R(n=' in line:
                print(line)
    
    print(f"SEED {seed} DONE")

print("\n=== ALL DONE ===")
