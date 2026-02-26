import time 
import math 
import heapq # cola de prioridad de A*, A* necesota sacar el nodo con menor coste  
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

# Intentamos usar scipy para inflar obstaculos con dilatacion. 
# Si no está, hacemos fallback para que el codigo funcione igual.
try:
    from scipy.ndimage import binary_dilation
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ============================================================
# CONFIG
# ============================================================
# Si True, ignora START/GOAL existentes y pide click en la imagen para elegirlos.
FORCE_INTERACTIVE = True 

MAP_IMAGE = "map_ours.png" 
SAVE_PREFIX = "pi3_result"

# Start/Goal en coordenadas de celda xy sobre el grid de la imagen.
START = (80, 520)   # (x, y) pixeles 
GOAL  = (520, 80)   # (x, y) pixeles

# Umbral para convertir en binario: 
# 1 (obstaculo) si intensidad < threshold, 0(libre) si >= threshold.
OBSTACLE_THRESHOLD = 0.75  

# Inflado de obstáculos para incluir la anchura del robot, pero en los obstaculos
ROBOT_RADIUS_CELLS = 6 

# True: A* permite movimientos en diagonal (8 vecinos).
# False: solo 4 direcciones (arriba/abajo/izq/der).
USE_8_CONNECTED = True

# Obstáculo dinámico: rectángulo con tamaño fijo (x_min, x_max, y_min, y_max)
# Lo colocamos "sobre el camino" automáticamente (más abajo).
DYN_OBS_SIZE = (25, 25)  # (ancho, alto) en celdas

# ============================================================
# UTILIDADES
# ============================================================

# Función para elegir START/GOAL
def pick_points_interactively(img_gray: np.ndarray, occ_inf: np.ndarray, n_points: int = 2):
    """
    Permite elegir puntos haciendo click sobre el mapa.
    Devuelve lista de puntos [(x,y), ...] en coordenadas de celda.
    """
    pts = []

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title("Haz click en START y luego en GOAL (solo en zona BLANCA/libre). Cierra la ventana al terminar.")
    ax.imshow(1 - occ_inf, cmap="gray", origin="upper")
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("y [cells]")

    def onclick(event):
        if event.inaxes != ax:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        # validar
        if not in_bounds(occ_inf, x, y):
            print(f"[click] ({x},{y}) fuera de mapa, elige otro")
            return
        if occ_inf[y, x] == 1:
            print(f"[click] ({x},{y}) está en obstáculo, elige otro")
            return

        pts.append((x, y))
        ax.scatter([x], [y], s=80, marker="o")
        fig.canvas.draw()
        print(f"[click] Punto {len(pts)}/{n_points}: {(x,y)}")
        if len(pts) >= n_points:
            fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    return pts

# 1) Cargar imagen y convertir a mapa de ocupación binario
def load_map_from_image(path: str) -> np.ndarray:
    """
    Convierte imagen de RGB -> float -> grayscale.
    """
    img = plt.imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)  # RGB->gris
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return img

# 2) Convertir imagen de grises a mapa de ocupación binario segun threshold
def image_to_occupancy(img_gray: np.ndarray, thr: float) -> np.ndarray:
    """
    Ocupado (obstáculo) = 1 si intensidad < thr.
    Libre = 0 si intensidad >= thr.
    """
    occ = (img_gray < thr).astype(np.uint8)
    return occ

# 3) Inflar obstáculos para tener margen de seguridad (radio robot).
def inflate_obstacles(occ: np.ndarray, radius_cells: int) -> np.ndarray:

    if radius_cells <= 0: # si el radio es menor que 0 no hace nada 
        return occ.copy()

    if SCIPY_OK:
        # Si ha hay SciPy pues crea una máscara circular (disco) con el radio para ensanchar
        rr = radius_cells
        yy, xx = np.ogrid[-rr:rr+1, -rr:rr+1]
        se = (xx*xx + yy*yy) <= rr*rr
        inflated = binary_dilation(occ.astype(bool), structure=se).astype(np.uint8)
        return inflated
    else:
        # Si no hay SciPy ensancha con un cuadrado
        k = radius_cells
        inflated = occ.copy()
        H, W = occ.shape
        ys, xs = np.where(occ == 1)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y-k), min(H, y+k+1)
            x0, x1 = max(0, x-k), min(W, x+k+1)
            inflated[y0:y1, x0:x1] = 1
        return inflated

# 4) chequeos de coordenadas
def in_bounds(occ: np.ndarray, x: int, y: int) -> bool:
    # Devuelve true si xy estan en el mapa
    H, W = occ.shape
    return 0 <= x < W and 0 <= y < H

def is_free(occ: np.ndarray, x: int, y: int) -> bool:
    # Devuelve true si xy estan en el mapa y además estan libres  
    return in_bounds(occ, x, y) and occ[y, x] == 0

# 5) vecinos en la rejilla y costes de moverse a cada vecino para A*
def neighbors(x: int, y: int, use_8: bool) -> List[Tuple[int, int, float]]:
    if use_8:
        dirs = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]
    else:
        dirs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
    return [(x+dx, y+dy, c) for dx, dy, c in dirs]

# 6)“A* usa esta estimación para ir ‘tirando’ hacia el objetivo sin explorar todo el mapa
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    # Euclídea
    return math.hypot(a[0]-b[0], a[1]-b[1])

# 7) reconstruyo el camino por donde he venido, guardando el padre de cada nodo
def reconstruct_path(came_from: Dict[Tuple[int,int], Tuple[int,int]],
                     cur: Tuple[int,int]) -> List[Tuple[int,int]]:
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path



# 8) __________________ ALGORITMO A* ____________________

def astar(occ: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int], use_8: bool = True) -> Optional[List[Tuple[int,int]]]:
    """
    A* clásico sobre grid binario.
    """
    if not is_free(occ, *start):
        print("[A*] START está en obstáculo o fuera.")
        return None
    if not is_free(occ, *goal):
        print("[A*] GOAL está en obstáculo o fuera.")
        return None

    open_heap = [] # cola de prioridad
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))

    came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
    gscore: Dict[Tuple[int,int], float] = {start: 0.0}
    closed = set()

    while open_heap:
        # g = lo que me ha costado llegar hasta aquí
        # h = lo que creo que falta hasta el objetivo
        # f = g + h = coste total estimado 
        f, g, cur = heapq.heappop(open_heap) # heapq siempre saca el nodo con menor f 
        if cur in closed:
            continue
        if cur == goal:
            return reconstruct_path(came_from, cur) # si hemos llegado reconstruyo

        closed.add(cur) # si no estamos en el objetivo, cierro el nodo para no volver a visitarlo
        cx, cy = cur
        for nx, ny, cost in neighbors(cx, cy, use_8): # vecinos del nodo actual, con su coste de movimiento
            if not is_free(occ, nx, ny):
                continue

            # Evitar cortar esquinas: si vas diagonal, exige libres los ortogonales.
            if use_8 and abs(nx-cx) == 1 and abs(ny-cy) == 1:
                if not (is_free(occ, nx, cy) and is_free(occ, cx, ny)):
                    continue
            
            # nuevo coste para llegar a vecino
            ng = g + cost 
            if ng < gscore.get((nx, ny), float("inf")):  # si es mejor que cualquier coste previo para ese vecino
                # actualizar coste y padre
                gscore[(nx, ny)] = ng
                came_from[(nx, ny)] = cur
                nf = ng + heuristic((nx, ny), goal)
                heapq.heappush(open_heap, (nf, ng, (nx, ny)))

    return None

#    ____________________________________________________


# 9) longitud de un camino (suma de distancias entre waypoints) para comparar inicial/replanificado
def path_length(path: List[Tuple[int,int]]) -> float:
    if len(path) < 2:
        return 0.0
    L = 0.0
    for (x0,y0),(x1,y1) in zip(path[:-1], path[1:]):
        L += math.hypot(x1-x0, y1-y0)
    return L

# 10) Funciones del obstaculo dinamico: calculo y aplico en el mapa
def pick_dynamic_obstacle_on_path(path: List[Tuple[int,int]], size_wh: Tuple[int,int]) -> Tuple[int,int,int,int]:
    """
    Elige un punto aproximadamente a mitad del path y crea un rectángulo ocupándolo.
    """
    w, h = size_wh
    idx = max(0, min(len(path)-1, len(path)//2))
    cx, cy = path[idx]
    x0 = cx - w//2
    x1 = cx + w//2
    y0 = cy - h//2
    y1 = cy + h//2
    return x0, x1, y0, y1

def apply_rect_obstacle(occ: np.ndarray, rect: Tuple[int,int,int,int]) -> np.ndarray:
    x0, x1, y0, y1 = rect
    occ2 = occ.copy()
    H, W = occ.shape
    x0c, x1c = max(0, x0), min(W-1, x1)
    y0c, y1c = max(0, y0), min(H-1, y1)
    occ2[y0c:y1c+1, x0c:x1c+1] = 1
    return occ2

# 11) Detecta el primer momento de colisión, una vez metido el obstaculo
def first_collision_index(occ: np.ndarray, path: List[Tuple[int,int]]) -> Optional[int]:
    """
    Devuelve el primer índice del path que cae en obstáculo.
    """
    for i, (x,y) in enumerate(path):
        if not is_free(occ, x, y):
            return i
    return None



# ============================================================
# MAIN
# ============================================================
def main():
    # 1) Cargar imagen y pasar a occupancy
    img = load_map_from_image(MAP_IMAGE)

    DOWNSAMPLE = 2  # reduce el mapa para que A* no tarde # NOTA: 2 -> mitad, 3 -> tercio
    img = img[::DOWNSAMPLE, ::DOWNSAMPLE]

    occ = image_to_occupancy(img, OBSTACLE_THRESHOLD)

    # Escalar parámetros al downsample
    robot_radius_cells = max(1, int(round(ROBOT_RADIUS_CELLS / DOWNSAMPLE)))
    dyn_obs_size = (
        max(3, int(round(DYN_OBS_SIZE[0] / DOWNSAMPLE))),
        max(3, int(round(DYN_OBS_SIZE[1] / DOWNSAMPLE))),
    )

    # 2) Inflar
    occ_inf = inflate_obstacles(occ, robot_radius_cells)

    # START/GOAL marcarles o usar los default
    start = START
    goal = GOAL

    need_pick = FORCE_INTERACTIVE or (not is_free(occ_inf, *start)) or (not is_free(occ_inf, *goal))

    if need_pick:
        print("[INFO] Selección interactiva activada (START y GOAL).")
        pts = pick_points_interactively(img, occ_inf, n_points=2)
        if len(pts) == 2:
            start, goal = pts[0], pts[1]
            print(f"[INFO] START={start} GOAL={goal}")
        else:
            raise RuntimeError("No se seleccionaron START/GOAL correctamente.")
    else:
        print(f"[INFO] START/GOAL por defecto: START={start} GOAL={goal}")

    # 3) A*
    t0 = time.perf_counter()
    path0 = astar(occ_inf, start, goal, use_8=USE_8_CONNECTED)
    t_plan0 = (time.perf_counter() - t0)

    if path0 is None:
        raise RuntimeError("No se encontró path inicial. Ajusta START/GOAL, threshold o inflado.")

    L0 = path_length(path0)

    # 4) Insertar obstáculo dinámico sobre el path
    dyn_rect = pick_dynamic_obstacle_on_path(path0, dyn_obs_size)
    occ_dyn = apply_rect_obstacle(occ_inf, dyn_rect)

    # 5) Simular avance por waypoints y detectar que se invalida
    # “Robot” avanza por el path inicial hasta chocar con el obstáculo dinámico.
    collision_i = first_collision_index(occ_dyn, path0)

    if collision_i is None:
        # No ha afectado al path (puede pasar si el obstáculo quedó fuera o el path lo bordeó)
        print("El obstáculo dinámico no corta la ruta. ")
        collision_i = len(path0) - 1

    # Pose actual = punto justo antes de la colisión (o al menos start)
    current_i = max(0, collision_i - 1)
    current_pose = path0[current_i]

    # 6) Replanificar desde pose actual al goal con el mapa actualizado
    t1 = time.perf_counter()
    path1 = astar(occ_dyn, current_pose, goal, use_8=USE_8_CONNECTED)
    t_plan1 = (time.perf_counter() - t1)

    if path1 is None:
        print("[A*] No hay ruta alternativa tras obstáculo dinámico. (Es posible si el bloqueo es total).")
        path1 = [current_pose]

    L1 = path_length(path1)

    # 7) Métricas
    metrics = {
        "plan_time_s_initial": t_plan0,
        "plan_time_s_replan": t_plan1,
        "path_len_cells_initial": L0,
        "path_len_cells_replan": L1,
        "collision_index_on_initial": collision_i,
        "start_original": START,
        "goal_original": GOAL,
        "start": start,
        "goal": goal,
        "current_pose_for_replan": current_pose,
        "robot_radius_cells": robot_radius_cells,
        "dyn_obs_size_cells": dyn_obs_size,
        "DOWNSAMPLE": DOWNSAMPLE,
        "map_shape_cells": occ.shape,
        "occ_ratio_dynamic": float(occ_dyn.mean()),
        "use_8_connected": USE_8_CONNECTED,
        "dyn_obstacle_rect": dyn_rect,
        "threshold": OBSTACLE_THRESHOLD,
    }

    # 8) Plots
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.set_title("PI-III: A* + obstáculo dinámico + replanificación")

    # dibujar mapa: libre blanco, obstáculo negro
    ax.imshow(1 - occ_dyn, cmap="gray", origin="upper")

    # path inicial (antes del obstáculo)
    px0 = [p[0] for p in path0]
    py0 = [p[1] for p in path0]
    ax.plot(px0, py0, linewidth=2, label="Path inicial (A*)")

    # marcar colisión
    ax.scatter([path0[collision_i][0]], [path0[collision_i][1]], s=60, marker="x", label="Colisión (ruta inválida)")

    # path replanteado
    px1 = [p[0] for p in path1]
    py1 = [p[1] for p in path1]
    ax.plot(px1, py1, linewidth=2, label="Path replanificado (A*)")

    # start/goal/current
    ax.scatter([start[0]], [start[1]], s=60, marker="o", label="Start")
    ax.scatter([goal[0]],  [goal[1]],  s=60, marker="*", label="Goal")
    ax.scatter([current_pose[0]], [current_pose[1]], s=60, marker="s", label="Pose replanificación")

    # obstáculo dinámico (rectángulo)
    x0, x1, y0, y1 = dyn_rect
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=2, label="Obstáculo dinámico")

    ax.legend()
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("y [cells]")

    fig.tight_layout()
    fig.savefig(f"{SAVE_PREFIX}_paths.png", dpi=200)

    # Guardar métricas a txt
    with open(f"{SAVE_PREFIX}_metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("=== OK ===")
    print(f"Figura guardada: {SAVE_PREFIX}_paths.png")
    print(f"Métricas guardadas: {SAVE_PREFIX}_metrics.txt")
    for k, v in metrics.items():
        print(f"{k}: {v}")


    plt.imsave(f"{SAVE_PREFIX}_occ_raw.png", 1 - occ, cmap="gray")
    plt.imsave(f"{SAVE_PREFIX}_occ_inflated.png", 1 - occ_inf, cmap="gray")
    plt.imsave(f"{SAVE_PREFIX}_occ_dynamic.png", 1 - occ_dyn, cmap="gray")

    plt.show()

    


if __name__ == "__main__":
    main()