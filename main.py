import os
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores


# =============================
# RUTAS FIJAS (WINDOWS)
# =============================
BAG_PATH = r"C:\Users\david\Downloads\material (1)\mir_basics_20251210_114529.bag"
OUT_DIR  = r"C:\Users\david\Downloads\material (1)\out_slam"


# =============================
# UTILIDADES SE(2)
# =============================
def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def yaw_from_quat_xyzw(qx, qy, qz, qw) -> float:
    # ROS quaternion order: x,y,z,w
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def se2_from_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]], dtype=np.float64)

def se2_inv(T: np.ndarray) -> np.ndarray:
    R = T[:2, :2]
    t = T[:2, 2]
    Ti = np.eye(3, dtype=np.float64)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ t
    return Ti


# =============================
# OCCUPANCY GRID (LOG-ODDS)
# =============================
@dataclass
class GridMap:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    logodds: np.ndarray

    @staticmethod
    def create(resolution: float = 0.05, size_m: float = 30.0, origin_at_center: bool = True) -> "GridMap":
        width = int(size_m / resolution)
        height = int(size_m / resolution)
        if origin_at_center:
            origin_x = - (width * resolution) / 2.0
            origin_y = - (height * resolution) / 2.0
        else:
            origin_x = 0.0
            origin_y = 0.0
        logodds = np.zeros((height, width), dtype=np.float32)
        return GridMap(resolution, width, height, origin_x, origin_y, logodds)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int((x - self.origin_x) / self.resolution)
        cy = int((y - self.origin_y) / self.resolution)
        return cx, cy

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    def update_ray(self, x0: float, y0: float, x1: float, y1: float,
                   lo_occ: float, lo_free: float):
        """
        Bresenham ray trace: cells along ray are free, endpoint is occupied.
        """
        cx0, cy0 = self.world_to_cell(x0, y0)
        cx1, cy1 = self.world_to_cell(x1, y1)

        if not self.in_bounds(cx0, cy0):
            return

        dx = abs(cx1 - cx0)
        dy = abs(cy1 - cy0)
        sx = 1 if cx0 < cx1 else -1
        sy = 1 if cy0 < cy1 else -1
        err = dx - dy

        cx, cy = cx0, cy0
        # mark free along the way (excluding endpoint)
        while (cx, cy) != (cx1, cy1):
            if self.in_bounds(cx, cy):
                self.logodds[cy, cx] += lo_free

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

            if not self.in_bounds(cx, cy):
                return

        # endpoint occupied
        if self.in_bounds(cx1, cy1):
            self.logodds[cy1, cx1] += lo_occ

    def prob(self) -> np.ndarray:
        l = np.clip(self.logodds, -20, 20)
        return 1.0 / (1.0 + np.exp(-l))


# =============================
# ESTRUCTURAS DE DATOS
# =============================
@dataclass
class TimedScan:
    t: float
    angle_min: float
    angle_inc: float
    ranges: np.ndarray
    range_min: float
    range_max: float

@dataclass
class TimedOdom:
    t: float
    x: float
    y: float
    yaw: float

@dataclass
class TimedMap:
    t: float
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    data: np.ndarray  # -1 unknown, 0..100 occupied


# =============================
# LECTURA DEL BAG (MIR)
# =============================
def read_bag_topics(bag_path: str) -> Dict[str, List]:
    """
    Lee topics estándar del MIR:
    /scan
    /odom
    /odometry/filtered
    /base_pose_ground_truth
    /amcl_pose
    /map
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)

    scans: List[TimedScan] = []
    odom: List[TimedOdom] = []
    odom_f: List[TimedOdom] = []
    gt: List[TimedOdom] = []
    amcl: List[TimedOdom] = []
    ref_maps: List[TimedMap] = []

    wanted = {
        "/scan",
        "/odom",
        "/odometry/filtered",
        "/base_pose_ground_truth",
        "/amcl_pose",
        "/map",
    }

    with Reader(bag_path) as reader:
        for conn, t_nsec, raw in reader.messages():
            if conn.topic not in wanted:
                continue

            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            t = t_nsec * 1e-9  # seconds

            if conn.topic == "/scan":
                ranges = np.array(msg.ranges, dtype=np.float32)
                scans.append(
                    TimedScan(
                        t=t,
                        angle_min=float(msg.angle_min),
                        angle_inc=float(msg.angle_increment),
                        ranges=ranges,
                        range_min=float(msg.range_min),
                        range_max=float(msg.range_max),
                    )
                )

            elif conn.topic in ("/odom", "/odometry/filtered", "/base_pose_ground_truth"):
                px = float(msg.pose.pose.position.x)
                py = float(msg.pose.pose.position.y)
                q = msg.pose.pose.orientation
                yaw = yaw_from_quat_xyzw(float(q.x), float(q.y), float(q.z), float(q.w))
                rec = TimedOdom(t=t, x=px, y=py, yaw=yaw)
                if conn.topic == "/odom":
                    odom.append(rec)
                elif conn.topic == "/odometry/filtered":
                    odom_f.append(rec)
                else:
                    gt.append(rec)

            elif conn.topic == "/amcl_pose":
                px = float(msg.pose.pose.position.x)
                py = float(msg.pose.pose.position.y)
                q = msg.pose.pose.orientation
                yaw = yaw_from_quat_xyzw(float(q.x), float(q.y), float(q.z), float(q.w))
                amcl.append(TimedOdom(t=t, x=px, y=py, yaw=yaw))

            elif conn.topic == "/map":
                res = float(msg.info.resolution)
                w = int(msg.info.width)
                h = int(msg.info.height)
                ox = float(msg.info.origin.position.x)
                oy = float(msg.info.origin.position.y)
                data = np.array(msg.data, dtype=np.int16).reshape(h, w)
                ref_maps.append(
                    TimedMap(t=t, resolution=res, width=w, height=h,
                             origin_x=ox, origin_y=oy, data=data)
                )

    scans.sort(key=lambda s: s.t)
    odom.sort(key=lambda o: o.t)
    odom_f.sort(key=lambda o: o.t)
    gt.sort(key=lambda o: o.t)
    amcl.sort(key=lambda o: o.t)
    ref_maps.sort(key=lambda m: m.t)

    return {
        "scans": scans,
        "odom": odom,
        "odom_f": odom_f,
        "gt": gt,
        "amcl": amcl,
        "ref_maps": ref_maps,
    }


# =============================
# INTERPOLACIÓN A TIEMPO t
# =============================
def interp_pose_at(t: float, traj: List[TimedOdom]) -> Optional[TimedOdom]:
    if len(traj) < 2:
        return None
    ts = [p.t for p in traj]
    if t < ts[0] or t > ts[-1]:
        return None

    # binary search for first index with traj[i].t >= t
    lo, hi = 0, len(traj) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if traj[mid].t < t:
            lo = mid + 1
        else:
            hi = mid - 1

    i1 = max(1, lo)
    i0 = i1 - 1
    p0, p1 = traj[i0], traj[i1]

    a = (t - p0.t) / (p1.t - p0.t + 1e-12)
    x = (1 - a) * p0.x + a * p1.x
    y = (1 - a) * p0.y + a * p1.y
    dyaw = wrap_pi(p1.yaw - p0.yaw)
    yaw = wrap_pi(p0.yaw + a * dyaw)

    return TimedOdom(t=t, x=float(x), y=float(y), yaw=float(yaw))


# =============================
# LaserScan -> puntos en frame robot
# =============================
def scan_to_points(scan: TimedScan, stride: int = 6, max_range: Optional[float] = None) -> np.ndarray:
    r_full = scan.ranges
    r = r_full[::stride].astype(np.float32)

    if max_range is None:
        max_range = scan.range_max

    idx = np.arange(0, len(r_full), stride, dtype=np.float32)
    ang = scan.angle_min + scan.angle_inc * idx

    mask = np.isfinite(r) & (r > scan.range_min) & (r < max_range)
    r = r[mask]
    ang = ang[mask]

    x = r * np.cos(ang)
    y = r * np.sin(ang)
    return np.stack([x, y], axis=1)


# =============================
# MÉTRICAS (ATE RMSE, YAW RMSE)
# =============================
def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")

def align_se2_by_first(est: List[TimedOdom], ref: List[TimedOdom]) -> List[TimedOdom]:
    """
    Alinea est a ref con un offset rígido usando la primera pose comparable.
    """
    if not est or not ref:
        return est

    # find first time where ref exists
    t0 = est[0].t
    r0 = interp_pose_at(t0, ref)
    if r0 is None:
        for e in est[1:]:
            r0 = interp_pose_at(e.t, ref)
            if r0 is not None:
                t0 = e.t
                break
    if r0 is None:
        return est

    e0 = interp_pose_at(t0, est)
    if e0 is None:
        return est

    T_ref0 = se2_from_xyyaw(r0.x, r0.y, r0.yaw)
    T_est0 = se2_from_xyyaw(e0.x, e0.y, e0.yaw)
    T_align = T_ref0 @ se2_inv(T_est0)

    out = []
    for e in est:
        Te = se2_from_xyyaw(e.x, e.y, e.yaw)
        Ta = T_align @ Te
        x = float(Ta[0, 2])
        y = float(Ta[1, 2])
        yaw = math.atan2(Ta[1, 0], Ta[0, 0])
        out.append(TimedOdom(t=e.t, x=x, y=y, yaw=yaw))
    return out

def compute_metrics_vs_gt(est: List[TimedOdom], gt: List[TimedOdom]) -> Dict[str, float]:
    if not est or not gt:
        return {"ate_rmse_m": float("nan"), "yaw_rmse_deg": float("nan"), "n_samples": 0.0}

    ate_list = []
    dyaw_list = []

    for e in est:
        g = interp_pose_at(e.t, gt)
        if g is None:
            continue
        dx = e.x - g.x
        dy = e.y - g.y
        ate_list.append(math.sqrt(dx * dx + dy * dy))
        dyaw_list.append(wrap_pi(e.yaw - g.yaw))

    ate = np.array(ate_list, dtype=np.float64)
    dyaw = np.array(dyaw_list, dtype=np.float64)

    return {
        "ate_rmse_m": rmse(ate),
        "yaw_rmse_deg": rmse(np.rad2deg(dyaw)),
        "n_samples": float(len(ate)),
    }


# =============================
# PIPELINE PRINCIPAL (OPCIÓN 1)
# =============================
def run_pipeline(
    bag_path: str,
    outdir: str,
    grid_res: float = 0.05,
    grid_size_m: float = 30.0,
    beam_stride_for_mapping: int = 6,
):
    os.makedirs(outdir, exist_ok=True)

    data = read_bag_topics(bag_path)
    scans: List[TimedScan] = data["scans"]
    odom: List[TimedOdom] = data["odom"]
    odom_f: List[TimedOdom] = data["odom_f"]
    gt: List[TimedOdom] = data["gt"]
    amcl: List[TimedOdom] = data["amcl"]
    ref_maps: List[TimedMap] = data["ref_maps"]

    if len(scans) < 2:
        raise RuntimeError("No hay suficientes mensajes en /scan.")
    if len(odom_f) < 2:
        raise RuntimeError("No hay suficientes mensajes en /odometry/filtered. Esta opción lo requiere.")

    # Crear mapa
    grid = GridMap.create(resolution=grid_res, size_m=grid_size_m, origin_at_center=True)

    # Log-odds config
    p_occ = 0.70
    p_free = 0.35
    lo_occ = math.log(p_occ / (1 - p_occ))
    lo_free = math.log(p_free / (1 - p_free))
    lo_min, lo_max = -10.0, 10.0

    # Trayectoria usada (pose) = /odometry/filtered muestreada en /scan
    used_pose: List[TimedOdom] = []
    used_scans: List[TimedScan] = []

    for s in scans:
        pf = interp_pose_at(s.t, odom_f)
        if pf is not None:
            used_pose.append(pf)
            used_scans.append(s)

    if len(used_pose) < 2:
        raise RuntimeError("No puedo interpolar /odometry/filtered en tiempos de /scan.")

    # Integración de scans en el mapa
    for pose, scan in tqdm(list(zip(used_pose, used_scans)), desc="Mapping (pose=/odometry/filtered)"):
        T_wb = se2_from_xyyaw(pose.x, pose.y, pose.yaw)
        rx, ry = pose.x, pose.y

        pts = scan_to_points(scan, stride=beam_stride_for_mapping)
        if pts.shape[0] < 10:
            continue

        pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
        pts_w = (T_wb @ pts_h.T).T[:, :2]

        for (x1, y1) in pts_w:
            grid.update_ray(rx, ry, float(x1), float(y1), lo_occ=lo_occ, lo_free=lo_free)

        grid.logodds[:] = np.clip(grid.logodds, lo_min, lo_max)

    # Guardar trayectoria usada
    traj_csv = os.path.join(outdir, "trajectory_used_odometry_filtered.csv")
    with open(traj_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw_rad"])
        for p in used_pose:
            w.writerow([p.t, p.x, p.y, p.yaw])

    # Métricas vs GT
    metrics_raw = compute_metrics_vs_gt(used_pose, gt)
    used_pose_aligned = align_se2_by_first(used_pose, gt)
    metrics_aligned = compute_metrics_vs_gt(used_pose_aligned, gt)

    metrics_path = os.path.join(outdir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("METRICS vs /base_pose_ground_truth\n")
        f.write("Trajectory used: /odometry/filtered sampled at /scan timestamps\n\n")
        f.write(f"Samples used: {int(metrics_raw.get('n_samples', 0))}\n\n")
        f.write("RAW (no alignment):\n")
        f.write(f"  ATE RMSE (m):     {metrics_raw['ate_rmse_m']:.6f}\n")
        f.write(f"  Yaw RMSE (deg):   {metrics_raw['yaw_rmse_deg']:.6f}\n\n")
        f.write("ALIGNED by first comparable pose:\n")
        f.write(f"  ATE RMSE (m):     {metrics_aligned['ate_rmse_m']:.6f}\n")
        f.write(f"  Yaw RMSE (deg):   {metrics_aligned['yaw_rmse_deg']:.6f}\n")

    # Guardar mapa generado
    prob = grid.prob()
    plt.figure(figsize=(8, 8))
    plt.imshow(1.0 - prob, origin="lower", cmap="gray")
    plt.title("Mapa generado (pose=/odometry/filtered, log-odds)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "map_ours.png"), dpi=200)
    plt.close()

    # Guardar mapa referencia (/map) si existe
    if ref_maps:
        ref = ref_maps[-1]
        ref_img = ref.data.astype(np.float32)
        ref_img[ref_img < 0] = 50.0
        ref_img = ref_img / 100.0
        plt.figure(figsize=(8, 8))
        plt.imshow(1.0 - ref_img, origin="lower", cmap="gray")
        plt.title("Mapa de referencia del bag (/map)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "map_ref.png"), dpi=200)
        plt.close()

    # Plot trayectorias (muestreadas a tiempos de /scan)
    def sample_at_times(source: List[TimedOdom], times: List[TimedOdom]) -> List[TimedOdom]:
        out = []
        for p in times:
            q = interp_pose_at(p.t, source)
            if q:
                out.append(q)
        return out

    odom_at = sample_at_times(odom, used_pose)
    gt_at = sample_at_times(gt, used_pose)
    amcl_at = sample_at_times(amcl, used_pose)

    def traj_xy(tr: List[TimedOdom]):
        if not tr:
            return None
        xs = np.array([p.x for p in tr])
        ys = np.array([p.y for p in tr])
        return xs, ys

    plt.figure(figsize=(8, 6))
    for name, tr in [
        ("odom", odom_at),
        ("odom_filtered (used)", used_pose),
        ("ground_truth", gt_at),
        ("amcl", amcl_at),
    ]:
        xy = traj_xy(tr)
        if xy is None:
            continue
        plt.plot(xy[0], xy[1], label=name)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Trayectorias (muestreadas a tiempos de /scan)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "trajectory.png"), dpi=200)
    plt.close()

    print("\nOK ✅")
    print(f"Bag:    {bag_path}")
    print(f"Outdir: {outdir}")
    print("Generado:")
    print(f" - {traj_csv}")
    print(f" - {metrics_path}")
    print(" - map_ours.png, map_ref.png (si existe /map), trajectory.png")


def main():
    run_pipeline(
        bag_path=BAG_PATH,
        outdir=OUT_DIR,
        grid_res=0.05,
        grid_size_m=30.0,
        beam_stride_for_mapping=6,
    )


if __name__ == "__main__":
    main()
