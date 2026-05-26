#!/usr/bin/env python3
"""
Clean navigation: go from current position to a fixed goal.
Uses gap-seeking + rectangular collision box for obstacle avoidance.
No button pressing, no arm control — pure walk.

Optional: integrate fixed_pose_validator to filter corner_localizer candidates
before navigation. Use --validate flag.
"""
import sys, os, time, math, subprocess, rospy, argparse
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from std_srvs.srv import SetBool

rospy.init_node('nav_to_fixed')

# ─── Config ───
GOAL_X = 8.36        # operator station 1
GOAL_Y = 0.0
MAX_SPEED = 0.06     # m/s
MAX_ANG = 0.20       # rad/s
BRAKE_DIST = 1.0     # emergency stop within this distance
COLLISION_BOX_FWD = 1.2   # box forward range
COLLISION_BOX_WIDE = 0.40 # box half-width
RATE_HZ = 5

vpub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rospy.sleep(1)

# ─── Auto gait ───
rospy.wait_for_service('/humanoid_auto_gait', timeout=10)
rospy.ServiceProxy('/humanoid_auto_gait', SetBool)(True)
rospy.sleep(3)

def norm_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

def stop(n=20):
    for _ in range(n): vpub.publish(Twist()); time.sleep(0.05)

def get_pose():
    o = rospy.wait_for_message('/odom', Odometry, timeout=2)
    q = o.pose.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return o.pose.pose.position.x, o.pose.pose.position.y, yaw

def scan_gap(scan):
    """Find widest gap in ±90°, return (front_min, steer_bias, min_left, min_right)."""
    ranges = scan.ranges
    n = len(ranges)
    ai = scan.angle_increment
    center = n // 2

    deg90 = int(math.radians(90) / ai)
    front_min = 30.0
    for i in range(max(0, center - deg90), min(n, center + deg90)):
        r = ranges[i]
        if scan.range_min < r < 30 and r < front_min:
            front_min = r

    GAP_THRESH = 2.0
    gaps = []
    in_gap, gap_start = False, max(0, center - deg90)
    end = min(n, center + deg90)
    for i in range(gap_start, end):
        r = ranges[i]
        is_free = r > GAP_THRESH or r < scan.range_min or r > 30
        if is_free and not in_gap:
            in_gap, gap_start = True, i
        elif not is_free and in_gap:
            in_gap = False
            gc = (gap_start + i - 1) // 2
            gaps.append((gap_start, i - 1, gc, (i - gap_start) * ai))
    if in_gap:
        gc = (gap_start + end - 1) // 2
        gaps.append((gap_start, end - 1, gc, (end - gap_start) * ai))

    if not gaps:
        return front_min, 0.0, 30.0, 30.0

    best = max(gaps, key=lambda g: g[3] - abs(g[2] - center) * ai * 0.3)
    gc, width = best[2], best[3]
    gap_angle = (gc - center) * ai

    urgency = 1.0
    if front_min < 2.5:
        urgency = 1.0 + (2.5 - front_min) / 2.5 * 3.0
    desperation = max(0.5, 3.0 - width)
    steer_bias = gap_angle * urgency * desperation

    deg180 = int(math.radians(180) / ai)
    min_left = min((ranges[i] for i in range(center + deg90, min(n, center + deg180)) if scan.range_min < ranges[i] < 30), default=30.0)
    min_right = min((ranges[i] for i in range(max(0, center - deg180), center - deg90)) if scan.range_min < ranges[i] < 30), default=30.0)

    return front_min, steer_bias, min_left, min_right

def collision_box(scan):
    """Check rectangular collision box ahead. Return True if obstacle inside."""
    try:
        ai = scan.angle_increment
        amin = scan.angle_min
        for idx, r in enumerate(scan.ranges):
            if r < 0.1 or r > 2.0:
                continue
            th = amin + idx * ai
            x = r * math.cos(th)
            y = r * math.sin(th)
            if 0.1 < x < COLLISION_BOX_FWD and abs(y) < COLLISION_BOX_WIDE:
                return True
    except:
        pass
    return False

# ─── Go to goal ───
# ─── Optional: FIX geometry validation before navigating ───
def run_fix_validation():
    """Run fixed_pose_validator to filter candidates. Returns True if valid."""
    from fixed_pose_validator import (
        get_real_candidates, validate_candidates_at_fix, Candidate
    )
    seed = int(time.time()) % 100
    cands, frames = get_real_candidates(seed, 1)
    if len(frames) < 3 or not cands:
        print("FIX_VALIDATOR: insufficient frames, fallback to raw nav")
        return True  # fallback — let raw nav try

    results = validate_candidates_at_fix(cands, frames)
    valid = [r for r in results if r.selected]
    if not valid:
        print("FIX_VALIDATOR: NO candidate passed both right_ok AND back_ok — FAIL")
        for r in results[:5]:
            print(f"  cand[{r.candidate.id}] R(ok={r.right.ok} n={r.right.count}) "
                  f"B(ok={r.back.ok} n={r.back.count}) score={r.fix_score:.1f}")
        return False

    best = valid[0]
    print(f"FIX_VALIDATOR: selected cand[{best.candidate.id}] "
          f"pos=({best.candidate.x:.3f},{best.candidate.y:.3f}) yaw={best.candidate.yaw_deg:.1f}°")
    return True

# Parse CLI
ap = argparse.ArgumentParser()
ap.add_argument('--validate', action='store_true', help='Run FIX geometry validation first')
args_cli, _ = ap.parse_known_args()

if args_cli.validate:
    if not run_fix_validation():
        print("FIX validation failed — aborting navigation")
        sys.exit(1)

# ─── Go to goal ───
x0, y0, _ = get_pose()
print(f"Start: ({x0:.2f}, {y0:.2f})  → Goal: ({GOAL_X:.1f}, {GOAL_Y:.1f})")

rate = rospy.Rate(RATE_HZ)
steps = 0
while not rospy.is_shutdown() and steps < 1500:
    steps += 1

    try:
        cx, cy, yaw = get_pose()
    except:
        rate.sleep()
        continue

    dist = math.hypot(GOAL_X - cx, GOAL_Y - cy)
    if dist < 0.5:
        print(f"Reached! dist={dist:.2f}m")
        break

    try:
        s = rospy.wait_for_message('/scan', LaserScan, timeout=0.3)
        fm, bias, ml, mr = scan_gap(s)
    except:
        stop(10)
        rate.sleep()
        continue

    # Emergency: obstacle in collision box
    if collision_box(s):
        ts = 0.0
        ang = 0.30 if bias > 0 else (-0.30 if ml > mr else 0.30)
        cmd = Twist(); cmd.linear.x = ts; cmd.angular.z = ang
        vpub.publish(cmd)
        rate.sleep()
        continue

    # Normal navigation
    desired_yaw = math.atan2(GOAL_Y - cy, GOAL_X - cx)
    ae = norm_angle(desired_yaw - yaw)

    sf = max(0.0, min(1.0, (fm - 0.3) / 1.7))
    ts = MAX_SPEED * sf
    ang = 0.4 * ae + bias * 1.5
    ang = max(-MAX_ANG, min(MAX_ANG, ang))

    if ang > 0 and ml < 0.3:
        ang = min(ang, 0.10)
    elif ang < 0 and mr < 0.3:
        ang = max(ang, -0.10)

    cmd = Twist()
    cmd.linear.x = ts
    cmd.angular.z = ang
    vpub.publish(cmd)
    rate.sleep()

stop()
print(f"DONE. Final: ({cx:.2f}, {cy:.2f})")
