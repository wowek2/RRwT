import glob
import os
import heapq
import numpy as np
import pandas as pd  # Pandas świetnie liczy średnie kroczące
import rasterio
from rasterio.merge import merge
from pyproj import Transformer
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates

# --- KONFIGURACJA ---
INPUT_FOLDER = "./data/numeryczne_modele_terenu"
OUTPUT_CSV = "dane_rzeki.csv"

START_GPS = (15.693167, 50.925917)
END_GPS   = (15.658306, 50.934583)

PENALTY_UPHILL = 1000.0
CROSS_SECTION_WIDTH = 500.0
STEP_METERS = 1.0

# Parametr wygładzania profilu podłużnego
ROLLING_WINDOW = 400  # Średnia z 10 punktów (ok. 10 metrów)

# ==========================================
# 1. ŁADOWANIE (BEZ ZMIAN)
# ==========================================
def load_data_raw(folder):
    files = glob.glob(os.path.join(folder, "*.asc"))
    if not files: raise FileNotFoundError("Brak plików ASC.")
    print(f"Łączenie {len(files)} plików...")
    srcs = [rasterio.open(f) for f in files]
    mosaic, trans = merge(srcs)
    data = mosaic[0].astype(float)
    data[data < -100] = np.nan 
    return data, trans

def get_pixel_coords(lon, lat, transform):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    x, y = transformer.transform(lon, lat)
    r, c = rasterio.transform.rowcol(transform, x, y)
    return int(r), int(c)

# ==========================================
# 2. A* (BEZ ZMIAN)
# ==========================================
def find_path_astar(data, start, end):
    rows, cols = data.shape
    queue = [(0, start)]
    costs = {start: 0}
    came_from = {}
    target = end
    visited = 0
    LIMIT = 5_000_000
    print("Szukanie ścieżki A*...")
    
    while queue:
        _, curr = heapq.heappop(queue)
        visited += 1
        if curr == target: break
        if visited > LIMIT: return None

        r, c = curr
        h_curr = data[r, c]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                h_next = data[nr, nc]
                if np.isnan(h_next): continue
                dist = np.sqrt(dr**2 + dc**2)
                diff = h_next - h_curr
                move_cost = dist + (diff * PENALTY_UPHILL if diff > 0 else 0)
                new_cost = costs[curr] + move_cost
                if (nr, nc) not in costs or new_cost < costs[(nr, nc)]:
                    costs[(nr, nc)] = new_cost
                    prio = new_cost + np.sqrt((target[0]-nr)**2 + (target[1]-nc)**2)
                    heapq.heappush(queue, (prio, (nr, nc)))
                    came_from[(nr, nc)] = curr

    if target not in came_from: return None
    path = []
    curr = target
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    return path[::-1]

# ==========================================
# 3. GEOMETRIA (Z DODANYM WYGŁADZANIEM)
# ==========================================
def geometry_pipeline(dem, path_pixels, transform):
    print("Przetwarzanie geometrii...")
    
    # 1. Splines (X, Y)
    rows, cols = zip(*path_pixels)
    xs_raw, ys_raw = rasterio.transform.xy(transform, rows, cols, offset='center')
    tck, u = splprep([xs_raw, ys_raw], s=len(path_pixels)*250)
    
    total_dist_est = np.sum(np.sqrt(np.diff(xs_raw)**2 + np.diff(ys_raw)**2))
    num_points = int(total_dist_est / STEP_METERS)
    u_new = np.linspace(0, 1, num_points)
    rx, ry = splev(u_new, tck)
    
    # Dystans
    dist_steps = np.sqrt(np.diff(rx)**2 + np.diff(ry)**2)
    dist_steps = np.insert(dist_steps, 0, 0)
    cumulative_dist = np.cumsum(dist_steps)
    
    # 2. Pobieranie SUROWEJ wysokości środka
    inv_transform = ~transform
    center_cols, center_rows = inv_transform * (rx, ry)
    center_coords = np.vstack((center_rows, center_cols))
    z_raw = map_coordinates(dem, center_coords, order=1, mode='nearest')
    
    # --- NOWOŚĆ: WYGŁADZANIE PROFILU PODŁUŻNEGO ---
    print(f"Obliczanie średniej kroczącej (okno={ROLLING_WINDOW})...")
    
    # Używamy Pandas do obsługi okna i krawędzi
    s_raw = pd.Series(z_raw)
    
    # center=True: średnia brana z [i-5, ... i ... i+4]
    # min_periods=1: na krawędziach policz średnią z tego co jest (nawet z 1 punktu)
    z_smooth = s_raw.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean().values
    # -----------------------------------------------

    # 3. Przekroje poprzeczne
    cross_sections = []
    offsets = np.linspace(-CROSS_SECTION_WIDTH, CROSS_SECTION_WIDTH, int(CROSS_SECTION_WIDTH*2) + 1)

    for i in range(len(rx)):
        p_prev = max(0, i-1)
        p_next = min(len(rx)-1, i+1)
        dx = rx[p_next] - rx[p_prev]
        dy = ry[p_next] - ry[p_prev]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0: length = 1
        nx, ny = -dy/length, dx/length
        
        cs_x_world = rx[i] + nx * offsets
        cs_y_world = ry[i] + ny * offsets
        cs_cols, cs_rows = inv_transform * (cs_x_world, cs_y_world)
        cs_coords = np.vstack((cs_rows, cs_cols))
        
        elevations = map_coordinates(dem, cs_coords, order=1, mode='nearest')
        cross_sections.append(elevations)
        
    return cumulative_dist, rx, ry, z_raw, z_smooth, np.array(cross_sections)

# ==========================================
# 4. EXPORT (ZAPISUJEMY OBIE WERSJE)
# ==========================================
def save_to_csv(filename, dists, rx, ry, z_raw, z_smooth, cs_data):
    print(f"Zapisywanie do {filename}...")
    
    data = {
        'dist_m': dists,
        'x_2180': rx,
        'y_2180': ry,
        'z_long_raw': z_raw,       # Wersja "poszarpana"
        'z_long_smooth': z_smooth  # Wersja wygładzona (10 pkt)
    }
    
    width = int((cs_data.shape[1] - 1) / 2)
    offsets_indices = range(-width, width+1)
    
    for i, offset in enumerate(offsets_indices):
        data[f"cs_{offset}"] = cs_data[:, i]
        
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print("Gotowe.")

if __name__ == "__main__":
    dem, trans = load_data_raw(INPUT_FOLDER)
    p_start = get_pixel_coords(*START_GPS, trans)
    p_end = get_pixel_coords(*END_GPS, trans)
    
    raw_path = find_path_astar(dem, p_start, p_end)
    
    if raw_path:
        # Rozpakowujemy dodatkową zmienną (z_smooth)
        dists, rx, ry, z_raw, z_smooth, cs_data = geometry_pipeline(dem, raw_path, trans)
        
        # Zapisujemy
        save_to_csv(OUTPUT_CSV, dists, rx, ry, z_raw, z_smooth, cs_data)
    else:
        print("Brak ścieżki.")