#!/usr/bin/env python3
"""
corner_shape_localizer.py v3 — Top-edge wall extraction + Hough multi-rho.
"""
import sys, os, time, math, numpy as np

FIX_X=5.01; FIX_Y=3.50; FIX_YAW=math.pi; NORTH_Y=4.50; EAST_X=6.25
RIGHT_TARGET=-1.00; BACK_TARGET=-1.24
MIN_R=0.35; MAX_R=20.0; VOXEL_SIZE=0.04; N_FRAMES=5
STRIP_WIDTH=0.12; MAX_MED_RESIDUAL=0.10; MAX_P90=0.20
MIN_RIGHT_COUNT=600; MIN_BACK_COUNT=500; MIN_FIX_SPAN=0.6
SPAWN_X=(3.7,5.9); SPAWN_Y=(2.1,3.9)
MAX_ORTH_ERR_DEG=8.0

GT_POSES={0:(4.31,2.27,-2.75),7:(5.11,2.48,-2.66),13:(4.62,3.45,1.678),31:(5.30,2.79,2.063),42:(5.46,2.70,-0.1185)}

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
def angle_diff(a,b): d=abs(wrap(a-b)); return min(d,abs(math.pi-d))
def clamp01(x): return max(0.,min(1.,x))

# ═══ Point cloud ═══

def collect_base_link_cloud(n_frames=N_FRAMES):
    import rospy; from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2; import tf2_ros, tf.transformations as tft
    b=tf2_ros.Buffer(); l=tf2_ros.TransformListener(b); rospy.sleep(0.5)
    tr,rm=None,None
    try:
        t=b.lookup_transform('base_link','radar',rospy.Time(0),rospy.Duration(2.0))
        tr=np.array([t.transform.translation.x,t.transform.translation.y,t.transform.translation.z])
        q=t.transform.rotation; rm=tft.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
    except: pass
    pts=[]
    for _ in range(n_frames):
        try:
            msg=rospy.wait_for_message('/lidar/points',PointCloud2,timeout=2.0)
            for p in pc2.read_points(msg,field_names=("x","y","z"),skip_nans=True): pts.append([p[0],p[1],p[2]])
            rospy.sleep(0.1)
        except: continue
    if not pts: return None
    a=np.array(pts)
    if rm is not None: a=a@rm.T+tr
    r=np.hypot(a[:,0],a[:,1]); return a[(r>MIN_R)&(r<MAX_R)]

# ═══ Voxel ═══

def deterministic_voxel_centroid(pts, voxel=0.03):
    if len(pts)<2: return pts
    k=np.floor(pts[:,:2]/voxel).astype(np.int64)
    _,inv=np.unique(k,axis=0,return_inverse=True)
    c=np.zeros((np.max(inv)+1,pts.shape[1]))
    np.add.at(c,inv,pts); cnt=np.bincount(inv)
    return c/cnt[:,None]

# ═══ Top-edge extraction ═══

def keep_top_z_per_angle_bin(points, angle_bin_deg=1.0, keep_ratio=0.20):
    x,y,z=points[:,0],points[:,1],points[:,2]
    ang=np.degrees(np.arctan2(y,x))
    bins=np.floor((ang+180.0)/angle_bin_deg).astype(int)
    parts=[]
    for b in np.unique(bins):
        idx=np.where(bins==b)[0]
        if len(idx)<5: continue
        cut=np.percentile(z[idx],100*(1.0-keep_ratio))
        parts.append(idx[z[idx]>=cut])
    if not parts: return np.empty((0,points.shape[1]))
    return points[np.concatenate(parts)]


def extract_wall_lines_v2(points):
    if points is None or len(points)<100: return [],[]
    x,y,z=points[:,0],points[:,1],points[:,2]; r=np.hypot(x,y)

    # 1. Basic filter
    base=points[(r>0.55)&(r<12.0)&np.isfinite(x)&np.isfinite(y)&np.isfinite(z)].copy()
    if len(base)<50: return [],[]

    # 2. Voxel down
    base=deterministic_voxel_centroid(base, voxel=0.03)

    # 3. Top-edge: global high-z
    z85=np.percentile(base[:,2],85); z90=np.percentile(base[:,2],90)
    top_main=base[base[:,2]>z85]
    # Near-wall supplement
    nr=base[(np.hypot(base[:,0],base[:,1])>0.35)&(np.hypot(base[:,0],base[:,1])<1.30)]
    near_top=np.empty((0,base.shape[1]))
    if len(nr)>50:
        nz70=np.percentile(nr[:,2],70)
        near_top=nr[nr[:,2]>nz70]

    wall_pts=np.vstack([top_main,near_top]) if len(top_main)>0 and len(near_top)>0 else (top_main if len(top_main)>0 else near_top)
    if len(wall_pts)<50: return [],[]
    wall_pts=deterministic_voxel_centroid(wall_pts, voxel=0.03)

    # 4. Angle-bin top-z filter (kill ring artifacts)
    wall_pts=keep_top_z_per_angle_bin(wall_pts, angle_bin_deg=1.0, keep_ratio=0.20)

    if len(wall_pts)<50: return [],[]
    return wall_pts, base


# ═══ Hough multi-rho ═══

def hough_multi_rho(xy, theta_step_deg=1.0, rho_step=0.03, min_votes=35, top_k=80):
    if len(xy)<20: return []
    al=np.deg2rad(np.arange(-180.,180.,theta_step_deg)); ca=np.cos(al); sa=np.sin(al)
    mr=float(np.max(np.linalg.norm(xy,axis=1)))+0.5; rb=np.arange(-mr,mr+rho_step,rho_step)
    acc=np.zeros((len(al),len(rb)-1),dtype=np.int32)
    for px,py in xy:
        rv=px*ca+py*sa; idx=np.floor((rv+mr)/rho_step).astype(np.int32)
        v=(idx>=0)&(idx<len(rb)-1); np.add.at(acc,(np.where(v)[0],idx[v]),1)
    peaks=[]
    for i in range(len(al)):
        row=acc[i].copy()
        for _ in range(6):
            j=int(np.argmax(row))
            if row[j]<min_votes: break
            rho=rb[j]+rho_step*0.5
            peaks.append({"alpha":float(al[i]),"rho":float(rho),"votes":int(row[j]),
                          "alpha_deg":math.degrees(al[i])%360})
            hw=max(1,int(0.15/rho_step)); lo=max(0,j-hw); hi=min(len(row),j+hw+1); row[lo:hi]=0
    # Sort by votes descending
    peaks.sort(key=lambda p:-p["votes"])
    # Dedup: angle<4° AND rho<0.20
    dedup=[]
    for p in peaks:
        if not any(angle_diff(p["alpha"],q["alpha"])<math.radians(4) and abs(p["rho"]-q["rho"])<0.20 for q in dedup):
            dedup.append(p)
        if len(dedup)>=top_k: break
    return dedup


# ═══ TLS refine ═══

def compute_line_span(points, alpha):
    if len(points)<2: return 0
    t=np.array([-math.sin(alpha),math.cos(alpha)])
    v=np.dot(points,t); return float(np.max(v)-np.min(v))

def refine_line_tls(xy, alpha0, rho0, dist_thresh=0.08, min_inliers=30):
    alpha,rho=alpha0,rho0
    for _ in range(3):
        c,s=math.cos(alpha),math.sin(alpha)
        d=np.abs(xy[:,0]*c+xy[:,1]*s-rho)
        inl=d<=dist_thresh
        if np.sum(inl)<min_inliers: return None
        pts=xy[inl]
        if len(pts)<3: return None
        # TLS: minimize sum (x cos a + y sin a - rho)^2
        mx=np.mean(pts[:,0]); my=np.mean(pts[:,1])
        dx=pts[:,0]-mx; dy=pts[:,1]-my
        cov=np.array([[np.sum(dx*dx),np.sum(dx*dy)],[np.sum(dx*dy),np.sum(dy*dy)]])
        w,v=np.linalg.eigh(cov)
        normal=v[:,0]  # smallest eigenvector = line normal
        alpha=math.atan2(normal[1],normal[0])
        if alpha<-math.pi: alpha+=2*math.pi
        if alpha>math.pi: alpha-=math.pi
        rho=mx*math.cos(alpha)+my*math.sin(alpha)
    c,s=math.cos(alpha),math.sin(alpha)
    d=np.abs(xy[:,0]*c+xy[:,1]*s-rho)
    inl=d<=dist_thresh
    n_inl=int(np.sum(inl))
    if n_inl<min_inliers: return None
    span=compute_line_span(xy[inl],alpha)
    return {"alpha":float(alpha),"rho":float(rho),"inliers":n_inl,"span":span,"points":xy[inl]}


# ═══ Full pipeline ═══

def detect_topedge_lines(points):
    wall_pts, base = extract_wall_lines_v2(points)
    if len(wall_pts)<50: return [],[],wall_pts

    peaks = hough_multi_rho(wall_pts[:,:2])
    if not peaks: return [],[],wall_pts

    lines = []
    for pk in peaks:
        ln = refine_line_tls(wall_pts[:,:2], pk["alpha"], pk["rho"])
        if ln is None: continue
        r_med = float(np.median(np.linalg.norm(ln["points"],axis=1)))
        # Quality gate
        if ln["inliers"]>=80 and ln["span"]>=0.40:
            ok=True
        elif r_med<1.30 and ln["inliers"]>=20 and ln["span"]>=0.20:
            ok=True
        else:
            continue
        lines.append({"alpha":ln["alpha"],"rho":ln["rho"],"inliers":ln["inliers"],
                      "span":ln["span"],"z_med":None,"r_med":r_med,"source":"topedge_hough",
                      "points":ln["points"]})

    # Orthogonal pairs
    pairs=[]
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            da=math.degrees(abs(angle_diff(lines[i]["alpha"],lines[j]["alpha"])))
            if abs(da-90.0)<=8:
                pairs.append((lines[i],lines[j]))
    return lines, pairs, wall_pts
