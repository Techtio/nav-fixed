#!/usr/bin/env python3
"""
topedge_corner_localizer v3 — Split Hough lines into continuous SEGMENTS,
find L-shapes by endpoint proximity, use directed rays for pose,
validate with full point cloud.

Core pipeline:
  top-edge pts → Hough → TLS → split into continuous segments
  → find L-shapes (endpoint-intersecting segments)
  → directed rays → pose → full-cloud validation → rank
"""

import sys, os, time, math, numpy as np

EAST_X=6.25; NORTH_Y=4.50; WORLD_CORNER=np.array([EAST_X,NORTH_Y])
SPAWN_X=(3.5,5.9); SPAWN_Y=(2.0,4.0)
MIN_R=0.35; MAX_R=20.0; N_FRAMES=5
VOXEL=0.04; TOPEDGE_RATIO=0.20; ANGLE_BIN=1.0
H_THETA=1.0; H_RHO=0.03; H_VOTES=30; H_TOPK=80
TLS_DIST=0.06; TLS_MIN=20
SEG_GAP=0.25; MIN_SEG_LEN_LONG=0.80; MIN_SEG_LEN_SHORT=0.25
MIN_INLIERS_LONG=60; MIN_INLIERS_SHORT=20; MAX_MED_RES=0.04; MAX_P90_RES=0.08; MIN_DENSITY=20
ORTH_DEG=8.0; CORNER_PROXIMITY=0.50; YAW_CONSISTENCY_DEG=8.0

GT_POSES={0:(4.31,2.27,-2.75),7:(5.11,2.48,-2.66),13:(4.62,3.45,1.678),31:(5.30,2.79,2.063),42:(5.46,2.70,-0.1185)}

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
def adiff_pi(a,b): return abs((a-b+math.pi/2)%math.pi-math.pi/2)
def adiff_deg(a,b): return math.degrees(adiff_pi(a,b))

# ═══ Point cloud ═══
def collect_base_link_cloud(n=N_FRAMES):
    import rospy; from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2; import tf2_ros, tf.transformations as tft
    b=tf2_ros.Buffer(); l=tf2_ros.TransformListener(b); rospy.sleep(0.5)
    tr,rm=None,None
    try:
        t=b.lookup_transform('base_link','radar',rospy.Time(0),rospy.Duration(2.))
        tr=np.array([t.transform.translation.x,t.transform.translation.y,t.transform.translation.z])
        q=t.transform.rotation; rm=tft.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
    except: pass
    pts=[]
    for _ in range(n):
        try:
            msg=rospy.wait_for_message('/lidar/points',PointCloud2,timeout=2.)
            for p in pc2.read_points(msg,field_names=("x","y","z"),skip_nans=True): pts.append([p[0],p[1],p[2]])
            rospy.sleep(0.1)
        except: continue
    if not pts: return None
    a=np.array(pts)
    if rm is not None: a=a@rm.T+tr
    r=np.hypot(a[:,0],a[:,1]); return a[(r>MIN_R)&(r<MAX_R)]

# ═══ Voxel + top-edge ═══
def vox_centroid(pts,v=VOXEL):
    if len(pts)<2: return pts
    k=np.floor(pts[:,:2]/v).astype(np.int64); _,iv=np.unique(k,axis=0,return_inverse=True)
    c=np.zeros((np.max(iv)+1,pts.shape[1])); np.add.at(c,iv,pts); return c/np.bincount(iv)[:,None]

def topz_per_angle(pts,ab=1.,kr=0.20):
    if len(pts)<10: return pts
    x,y,z=pts[:,0],pts[:,1],pts[:,2]
    ang=np.degrees(np.arctan2(y,x)); bins=np.floor((ang+180.)/ab).astype(int)
    parts=[]
    for b in np.unique(bins):
        idx=np.where(bins==b)[0]
        if len(idx)<5: continue
        cut=np.percentile(z[idx],100*(1.-kr)); parts.append(idx[z[idx]>=cut])
    return np.empty((0,pts.shape[1])) if not parts else pts[np.concatenate(parts)]

def extract_topedge(pts):
    if pts is None or len(pts)<100: return np.empty((0,3))
    x,y,z=pts[:,0],pts[:,1],pts[:,2]; r=np.hypot(x,y)
    base=pts[(r>0.55)&(r<12.)&np.isfinite(x)&np.isfinite(y)&np.isfinite(z)]
    if len(base)<50: return np.empty((0,3))
    base=vox_centroid(base,0.03)
    z85=np.percentile(base[:,2],85); top=base[base[:,2]>z85]
    nr=base[(np.hypot(base[:,0],base[:,1])>0.35)&(np.hypot(base[:,0],base[:,1])<1.30)]
    if len(nr)>50: nz70=np.percentile(nr[:,2],70); near=nr[nr[:,2]>nz70]; wall=np.vstack([top,near])
    else: wall=top
    if len(wall)<20: return np.empty((0,3))
    wall=vox_centroid(wall,0.03); wall=topz_per_angle(wall,ANGLE_BIN,TOPEDGE_RATIO)
    return wall

# ═══ Hough ═══
def hough_peaks(xy):
    if len(xy)<20: return []
    al=np.deg2rad(np.arange(-180.,180.,H_THETA)); ca=np.cos(al); sa=np.sin(al)
    mr=float(np.max(np.linalg.norm(xy,axis=1)))+0.5; rb=np.arange(-mr,mr+H_RHO,H_RHO)
    acc=np.zeros((len(al),len(rb)-1),dtype=np.int32)
    for px,py in xy:
        rv=px*ca+py*sa; idx=np.floor((rv+mr)/H_RHO).astype(np.int32)
        v=(idx>=0)&(idx<len(rb)-1); np.add.at(acc,(np.where(v)[0],idx[v]),1)
    peaks=[]
    for i in range(len(al)):
        row=acc[i].copy()
        for _ in range(6):
            j=int(np.argmax(row))
            if row[j]<H_VOTES: break
            peaks.append({"alpha":float(al[i]),"rho":float(rb[j]+H_RHO*.5),"votes":int(row[j])})
            hw=max(1,int(.15/H_RHO)); row[max(0,j-hw):min(len(row),j+hw+1)]=0
    peaks.sort(key=lambda p:-p["votes"])
    dedup=[]
    for p in peaks:
        if not any(adiff_pi(p["alpha"],q["alpha"])<np.deg2rad(4) and abs(p["rho"]-q["rho"])<.2 for q in dedup):
            dedup.append(p)
        if len(dedup)>=H_TOPK: break
    return dedup

# ═══ TLS + line segment ═══
def tls_refit(xy):
    if len(xy)<3: return None
    c=xy.mean(0); X=xy-c
    try: _,_,vh=np.linalg.svd(X,full_matrices=False)
    except: return None
    tg=vh[0]; n=np.array([-tg[1],tg[0]]); rho=float(n@c); a=float(np.arctan2(n[1],n[0]))
    if rho<0: rho=-rho; a+=math.pi
    a=(a+math.pi)%math.pi
    return {"alpha":a,"rho":rho,"tangent":tg,"normal":n}

def collect_inliers(xy,alpha,rho,th=TLS_DIST):
    c,s=math.cos(alpha),math.sin(alpha); d=np.abs(xy[:,0]*c+xy[:,1]*s-rho); return xy[d<=th]

def split_into_segments(pts, tangent):
    if len(pts)<2: return [pts]
    s=pts@tangent; o=np.argsort(s); ss=s[o]; pso=pts[o]
    segs=[]; start=0
    for i in range(1,len(ss)):
        if ss[i]-ss[i-1]>SEG_GAP: segs.append(pso[start:i]); start=i
    segs.append(pso[start:]); return segs

def segment_metrics(seg, line):
    if len(seg)==0: return None
    s=seg@line["tangent"]; length=float(np.max(s)-np.min(s))
    c,s=math.cos(line["alpha"]),math.sin(line["alpha"])
    res=np.abs(seg[:,0]*c+seg[:,1]*s-line["rho"])
    density=len(seg)/max(length,1e-6)
    return {"length":length,"count":len(seg),"density":density,
            "med_res":float(np.median(res)),"p90_res":float(np.percentile(res,90)),
            "points":seg,"tangent":line["tangent"],"normal":line["normal"],
            "alpha":line["alpha"],"rho":line["rho"]}

def segment_endpoints(seg_m):
    s=seg_m["points"]@seg_m["tangent"]; lo,hi=np.argmin(s),np.argmax(s)
    return seg_m["points"][lo][:2], seg_m["points"][hi][:2]

# ═══ L-shape ═══
def is_valid_l_shape(sa, sb):
    ang=abs(adiff_deg(sa["alpha"],sb["alpha"]))
    if abs(ang-90.)>ORTH_DEG: return False
    # Intersection
    c1,s1=math.cos(sa["alpha"]),math.sin(sa["alpha"]); c2,s2=math.cos(sb["alpha"]),math.sin(sb["alpha"])
    det=c1*s2-c2*s1
    if abs(det)<1e-10: return False
    cx=(sa["rho"]*s2-sb["rho"]*s1)/det; cy=(c1*sb["rho"]-c2*sa["rho"])/det
    corner=np.array([cx,cy])
    # Corner near endpoints
    ep_a=segment_endpoints(sa); ep_b=segment_endpoints(sb)
    da=min(np.linalg.norm(corner-ep_a[0]),np.linalg.norm(corner-ep_a[1]))
    db=min(np.linalg.norm(corner-ep_b[0]),np.linalg.norm(corner-ep_b[1]))
    if da>CORNER_PROXIMITY or db>CORNER_PROXIMITY: return False
    # At least one long
    if max(sa["length"],sb["length"])<MIN_SEG_LEN_LONG: return False
    if min(sa["length"],sb["length"])<MIN_SEG_LEN_SHORT: return False
    return True

# ═══ Directed rays + pose ═══
def far_endpoint(seg_m, corner):
    ep0,ep1=segment_endpoints(seg_m)
    d0=np.linalg.norm(ep0-corner); d1=np.linalg.norm(ep1-corner)
    return ep1 if d1>d0 else ep0

def solve_pose_from_directed(corner, ray_a, ray_b, assign_west, assign_south):
    """Given rays in base_link, map to world west (-1,0) and south (0,-1)."""
    west_world=np.array([-1.,0.])
    south_world=np.array([0.,-1.])
    if assign_west=="a": rw=ray_a; rs=ray_b
    else: rw=ray_b; rs=ray_a
    # yaw from aligning rays to world
    yaw_w=math.atan2(rw[1],rw[0])-math.pi  # align to west world
    yaw_s=math.atan2(rs[1],rs[0])+math.pi/2 # align to south world
    yaw_avg=wrap((yaw_w+yaw_s)/2.); yaw_c=abs(wrap(yaw_w-yaw_s))
    if yaw_c>np.deg2rad(YAW_CONSISTENCY_DEG): return None
    c,s=math.cos(yaw_avg),math.sin(yaw_avg); R=np.array([[c,-s],[s,c]])
    t=WORLD_CORNER-R@corner
    wx,wy=float(t[0]),float(t[1])
    if not (SPAWN_X[0]<=wx<=SPAWN_X[1] and SPAWN_Y[0]<=wy<=SPAWN_Y[1]): return None
    return (wx,wy,yaw_avg,yaw_c)

# ═══ Full cloud validation ═══
def validate_with_full_cloud(pose, pts_full):
    if len(pts_full)<100: return -1e9
    x,y,yaw=pose; c,s=math.cos(yaw),math.sin(yaw)
    pw=pts_full.copy(); pw[:,0]=pts_full[:,0]*c-pts_full[:,1]*s+x; pw[:,1]=pts_full[:,0]*s+pts_full[:,1]*c+y
    east=pw[np.abs(pw[:,0]-EAST_X)<.15]; north=pw[np.abs(pw[:,1]-NORTH_Y)<.15]
    if len(east)<30 or len(north)<30: return -1e9
    er=float(np.median(np.abs(east[:,0]-EAST_X))); nr=float(np.median(np.abs(north[:,1]-NORTH_Y)))
    eir=len(east)/len(pw); nir=len(north)/len(pw)
    sc=0; sc+=50.*eir+50.*nir-30.*er-30.*nr
    return sc

# ═══ Main ═══
def localize_by_topedge_lshape(pts_full):
    if pts_full is None or len(pts_full)<100: return [],{"stage":"no_points"}
    pts_filter=pts_full[np.hypot(pts_full[:,0],pts_full[:,1])<12.]
    edge_pts=extract_topedge(pts_filter)
    diag={"stage":"ok","n_full":len(pts_full),"n_topedge":len(edge_pts)}

    if len(edge_pts)<80: diag["stage"]="topedge_too_few"; return [],diag

    # Hough → TLS → dedup
    peaks=hough_peaks(edge_pts[:,:2])
    if not peaks: diag["stage"]="no_hough"; return [],diag

    lines=[]
    for pk in peaks:
        inl=collect_inliers(edge_pts[:,:2],pk["alpha"],pk["rho"])
        if len(inl)<TLS_MIN: continue
        tls=tls_refit(inl)
        if tls is None: continue
        inl2=collect_inliers(edge_pts[:,:2],tls["alpha"],tls["rho"])
        if len(inl2)<TLS_MIN: continue
        lines.append({"tls":tls,"inliers":inl2})

    # Dedup
    lines.sort(key=lambda l:len(l["inliers"]),reverse=True)
    kept=[]
    for ln in lines:
        lt=ln["tls"]
        if not any(adiff_pi(lt["alpha"],k["tls"]["alpha"])<np.deg2rad(3) and abs(lt["rho"]-k["tls"]["rho"])<.08 for k in kept):
            kept.append(ln)
        if len(kept)>=20: break

    # Split into segments
    segments=[]
    sid=0
    for ln in kept:
        segs=split_into_segments(ln["inliers"],ln["tls"]["tangent"])
        for seg in segs:
            m=segment_metrics(seg,ln["tls"])
            if m is None: continue
            r=np.hypot(seg[:,0],seg[:,1]); is_near=float(np.median(r))<1.5
            min_len=MIN_SEG_LEN_SHORT if is_near else MIN_SEG_LEN_LONG
            min_inl=MIN_INLIERS_SHORT if is_near else MIN_INLIERS_LONG
            if m["length"]<min_len or m["count"]<min_inl: continue
            if m["med_res"]>MAX_MED_RES or m["p90_res"]>MAX_P90_RES: continue
            if m["density"]<MIN_DENSITY: continue
            m["id"]=sid; sid+=1; segments.append(m)
    diag["n_segments"]=len(segments)
    if len(segments)<2: diag["stage"]="too_few_segments"; return [],diag

    # Find L-shapes
    lshapes=[]
    for i,sa in enumerate(segments):
        for j in range(i+1,len(segments)):
            sb=segments[j]
            if is_valid_l_shape(sa,sb):
                c1,s1=math.cos(sa["alpha"]),math.sin(sa["alpha"])
                c2,s2=math.cos(sb["alpha"]),math.sin(sb["alpha"])
                det=c1*s2-c2*s1
                cx=(sa["rho"]*s2-sb["rho"]*s1)/det; cy=(c1*sb["rho"]-c2*sa["rho"])/det
                lshapes.append({"sa":sa,"sb":sb,"corner":np.array([cx,cy])})
    diag["n_lshapes"]=len(lshapes)
    if not lshapes: diag["stage"]="no_lshapes"; return [],diag

    # Generate candidates
    candidates=[]
    for ls in lshapes:
        sa,sb,corner=ls["sa"],ls["sb"],ls["corner"]
        ra=far_endpoint(sa,corner)-corner; ra=ra/np.linalg.norm(ra)
        rb=far_endpoint(sb,corner)-corner; rb=rb/np.linalg.norm(rb)
        for a_w,a_s in [("a","b"),("b","a")]:
            pose=solve_pose_from_directed(corner,ra,rb,a_w,a_s)
            if pose is None: continue
            sc=validate_with_full_cloud(pose,pts_filter)
            if sc<-500: continue
            candidates.append({"pose":pose,"full_score":sc,"sa":sa,"sb":sb,"corner":corner})
    diag["n_candidates"]=len(candidates)
    if not candidates: diag["stage"]="no_valid_pose"; return [],diag

    # Rank by full-cloud validation
    for c in candidates:
        c["score"]=c["full_score"]+min(c["sa"]["length"],2.)*3.+min(c["sb"]["length"],2.)*3.
        c["score"]+=c["sa"]["density"]*.1+c["sb"]["density"]*.1
    candidates.sort(key=lambda c:-c["score"])
    return candidates,diag
