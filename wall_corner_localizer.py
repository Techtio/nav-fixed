#!/usr/bin/env python3
"""
wall_corner_localizer.py v3 — Stabilized collection + per-frame QA + adaptive z.
"""
import sys, os, time, math, numpy as np

EAST_X=6.25; NORTH_Y=4.50
SPAWN_X=(3.8,5.8); SPAWN_Y=(2.0,3.95)
GT_POSES={0:(4.31,2.27,-2.75),7:(5.11,2.48,-2.66),13:(4.62,3.45,1.678),31:(5.30,2.79,2.063),42:(5.46,2.70,-0.1185)}

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

# ═══ Stabilized collection (per-frame QA + adaptive z) ═══

def stabilize_robot():
    """Zero cmd_vel, wait for robot to settle."""
    try:
        import rospy; from geometry_msgs.msg import Twist
        pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10); rospy.sleep(0.5)
        for _ in range(5): pub.publish(Twist()); rospy.sleep(0.1)
        rospy.sleep(3)
    except: time.sleep(3)

def collect_quality_frames(n_target=5, n_max=10):
    """Collect up to n_max frames, keep best n_target by quality metrics."""
    import rospy; from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2; import tf2_ros, tf.transformations as tft

    tf_buffer=tf2_ros.Buffer(); tf_listener=tf2_ros.TransformListener(tf_buffer); rospy.sleep(0.5)
    trans,rm=None,None
    try:
        t=tf_buffer.lookup_transform('base_link','radar',rospy.Time(0),rospy.Duration(3.))
        trans=np.array([t.transform.translation.x,t.transform.translation.y,t.transform.translation.z])
        q=t.transform.rotation; rm=tft.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
    except: pass

    frames=[]  # (pts_xyz, quality_score, diagnostics)
    for fid in range(n_max):
        diag={"frame_id":fid,"used":False,"n_points":0,"z_p5":None,"z_p50":None,"z_p95":None,
              "r_p50":None,"r_p90":None,"hough_raw":0}
        try:
            msg=rospy.wait_for_message('/lidar/points',PointCloud2,timeout=3.)
            pts=[np.array([p[0],p[1],p[2]]) for p in pc2.read_points(msg,field_names=("x","y","z"),skip_nans=True)]
            if not pts: diag["n_points"]=0; frames.append((None,0,diag)); rospy.sleep(0.15); continue
            a=np.array(pts)
            if rm is not None: a=a@rm.T+trans
            r=np.hypot(a[:,0],a[:,1])
            a=a[(r>0.35)&(r<20.)]
            diag["n_points"]=len(a)
            if len(a)<500: frames.append((None,0,diag)); rospy.sleep(0.15); continue
            diag["z_p5"]=round(float(np.percentile(a[:,2],5)),3)
            diag["z_p50"]=round(float(np.percentile(a[:,2],50)),3)
            diag["z_p95"]=round(float(np.percentile(a[:,2],95)),3)
            diag["r_p50"]=round(float(np.percentile(r,50)),3)
            diag["r_p90"]=round(float(np.percentile(r,90)),3)
            # Quick Hough check
            xy=a[:,:2][::10]  # subsample for speed
            try:
                from collections import Counter
                ang_bin=2; al_bins={}
                for deg in np.arange(-180,180,ang_bin):
                    a_=math.radians(float(deg)); rv=xy[:,0]*math.cos(a_)+xy[:,1]*math.sin(a_)
                    h,_=np.histogram(rv,bins=100); al_bins[deg]=int(np.max(h))
                diag["hough_raw"]=max(al_bins.values()) if al_bins else 0
            except: diag["hough_raw"]=0
            # Quality score: enough points, z spread, Hough signal
            qs=0.0; qs+=min(len(a)/50000.,1.)*5.; qs+=min((diag["z_p95"]-diag["z_p5"])/0.5,1.)*3.; qs+=min(diag["hough_raw"]/200.,1.)*5.
            frames.append((a,qs,diag))
        except: frames.append((None,0,diag))
        rospy.sleep(0.15)

    # Mark top n_target by quality
    valid=[(i,f) for i,f in enumerate(frames) if f[0] is not None and len(f[0])>=500]
    if not valid: return None,[]
    valid.sort(key=lambda x:-x[1][1])
    kept=[v[0] for v in valid[:n_target]]
    for i,f in enumerate(frames):
        if i in kept: frames[i][2]["used"]=True
    merged=np.vstack([frames[i][0] for i in kept])
    return merged,[frames[i][2] for i in range(len(frames))]

# ═══ Adaptive z filter ═══
def adaptive_z_filter(pts):
    """Use percentile-based z filter, not fixed z_max=0.1."""
    if pts is None or len(pts)<100: return pts
    z=pts[:,2]; z5=np.percentile(z,5); z95=np.percentile(z,95)
    m=(z>=z5)&(z<=z95)
    if m.sum()<50: return pts  # keep all if too few
    return pts[m]

# ═══ Multi-band Hough (from adaptive_frontend) ═══
def det_voxel(p,v=0.04):
    if len(p)<2: return p
    q=np.floor(p[:,:3]/v).astype(np.int64)
    o=np.lexsort((q[:,2],q[:,1],q[:,0])); p=p[o]; q=q[o]
    out=[]; i=0; n=len(p)
    while i<n:
        j=i+1
        while j<n and np.all(q[j]==q[i]): j+=1
        out.append(p[i:j].mean(0)); i=j
    return np.array(out,dtype=np.float64)

def _da180(a,b): return abs(((a-b+90)%180)-90)

def build_point_sets(pts, min_r=0.35, max_r=12.0, voxel=0.04):
    if pts is None or len(pts)<50: return []
    p=np.c_[pts[:,:3],np.linalg.norm(pts[:,:3],axis=1)]; x,y,z,r=p.T
    fin=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)&np.isfinite(r)
    base=fin&(r>=min_r)&(r<=max_r)
    if int(base.sum())<50: return []
    zb=z[base]; z5,z10,z30,z50,z70,z90,z95=np.percentile(zb,[5,10,30,50,70,90,95])
    masks=[
        ("old_10",base&(z>-1.5)&(z<0.10)),("wide_30",base&(z>-1.5)&(z<0.30)),
        ("wide_60",base&(z>-1.5)&(z<0.60)),("a_p05p95",base&(z>=z5)&(z<=z95)),
        ("a_p10p90",base&(z>=z10)&(z<=z90)),("low_p30",base&(z<=z30)),
        ("high_p70",base&(z>=z70)),("vlow_p10",base&(z<=z10)),("vhigh_p90",base&(z>=z90)),
        ("near_all",base&(r<=1.3)&(z>=z5)&(z<=z95)),
        ("near_low",base&(r<=1.3)&(z<=z50)),("near_high",base&(r<=1.3)&(z>=z50)),
    ]
    sets=[]; seen=set()
    for nm,m in masks:
        if int(m.sum())<40: continue
        ps=det_voxel(p[m,:3],voxel)
        if len(ps)<25: continue
        k=(len(ps),round(float(np.median(ps[:,2])),3))
        if k in seen: continue
        seen.add(k); sets.append((nm,ps))
    return sets

def hough_signed(pts, angle_step=1., rho_step=0.02, min_votes=15, top_k=80):
    p=np.asarray(pts,dtype=np.float64)
    if len(p)<max(min_votes,20): return []
    xy=p[:,:2]; mr=max(2.,float(np.max(np.linalg.norm(xy,axis=1)))+.5)
    rb=np.arange(-mr,mr+rho_step,rho_step); raw=[]
    for deg in np.arange(-180.,180.,angle_step):
        a=math.radians(float(deg)); rv=xy[:,0]*math.cos(a)+xy[:,1]*math.sin(a)
        h,ed=np.histogram(rv,bins=rb)
        for idx in np.flatnonzero(h>=min_votes):
            raw.append({"alpha":a,"rho":float((ed[idx]+ed[idx+1])*.5),"votes":int(h[idx])})
    raw.sort(key=lambda d:d["votes"],reverse=True)
    out=[]
    for ln in raw:
        ad=math.degrees(ln["alpha"]); rho=ln["rho"]
        if not any(_da180(ad,math.degrees(o["alpha"]))<1.5 and abs(rho-o["rho"])<.06 for o in out):
            out.append(ln)
        if len(out)>=top_k: break
    return out

def merge_lines(lns, angle_tol=2., rho_tol=0.08, top_k=120):
    lns=sorted(lns,key=lambda d:d.get("votes",0),reverse=True); m=[]
    for ln in lns:
        ad=math.degrees(ln["alpha"]); rho=ln["rho"]
        if not any(_da180(ad,math.degrees(o["alpha"]))<angle_tol and abs(rho-o["rho"])<rho_tol for o in m):
            m.append(dict(ln))
        if len(m)>=top_k: break
    return m

def detect_lines(pts):
    sets=build_point_sets(pts); all_=[]
    for nm,ps in sets:
        mv=max(15,int(len(ps)*0.002)); ln=hough_signed(ps,min_votes=mv)
        for l in ln: l["sources"]={nm}
        all_.extend(ln)
    return merge_lines(all_)

# ═══ Line variants + pose ═══
def line_variants(alpha,rho): return [(alpha,rho),(wrap(alpha+math.pi),-rho)]

def solve_pose(ea,er,na,nr):
    ye=wrap(-ea); yn=wrap(math.pi/2-na); yc=abs(math.degrees(wrap(ye-yn)))
    if yc>8.: return None
    yw=math.atan2(math.sin(ye)+math.sin(yn),math.cos(ye)+math.cos(yn))
    x=EAST_X-er; y=NORTH_Y-nr
    if not (SPAWN_X[0]<=x<=SPAWN_X[1] and SPAWN_Y[0]<=y<=SPAWN_Y[1]): return None
    return {"x":x,"y":y,"yaw":wrap(yw),"yaw_consistency_deg":yc}

def world_residual(pts,pose,band=0.08):
    x,y,yaw=pose["x"],pose["y"],pose["yaw"]; c,s=math.cos(yaw),math.sin(yaw)
    X=c*pts[:,0]-s*pts[:,1]+x; Y=s*pts[:,0]+c*pts[:,1]+y
    de=np.abs(X-EAST_X); dn=np.abs(Y-NORTH_Y); me=de<band; mn=dn<band
    ec=int(me.sum()); nc=int(mn.sum())
    em=float(np.median(de[me])) if ec>20 else 9.0
    nm=float(np.median(dn[mn])) if nc>20 else 9.0
    return ec,em,nc,nm

# ═══ Main ═══
def corner_localize(pts):
    if pts is None or len(pts)<100: return [],[],{}
    pts_f=adaptive_z_filter(pts)
    peaks=detect_lines(pts_f)
    if not peaks: return [],[],{"n_peaks":0}
    r=np.hypot(pts_f[:,0],pts_f[:,1])
    P=pts_f[(r>0.35)&(r<12.)&(pts_f[:,2]>-1.5)&(pts_f[:,2]<0.6)]
    if len(P)<50: P=pts_f
    cands=[]
    for i in range(len(peaks)):
        for j in range(i+1,len(peaks)):
            ai,ri=peaks[i]["alpha"],peaks[i]["rho"]; aj,rj=peaks[j]["alpha"],peaks[j]["rho"]
            da=abs((math.degrees(ai-aj)+90)%180-90)
            if abs(da-90)>6: continue
            for ae,re in line_variants(ai,ri):
                for an,rn in line_variants(aj,rj):
                    for aeast,reast,anorth,rnorth,ident in [(ae,re,an,rn,"A_east_B_north"),(an,rn,ae,re,"B_east_A_north")]:
                        sol=solve_pose(aeast,reast,anorth,rnorth)
                        if sol is None: continue
                        ec,em,nc,nm=world_residual(P,sol)
                        sc=-(em+nm)+0.0002*(ec+nc)-0.01*sol["yaw_consistency_deg"]
                        cands.append({**sol,"score":sc,"identity":ident,"line_i":i,"line_j":j,"east_count":ec,"east_med":em,"north_count":nc,"north_med":nm})
    cands.sort(key=lambda d:d["score"],reverse=True)
    return cands,peaks,{"n_peaks":len(peaks),"n_cands":len(cands)}
