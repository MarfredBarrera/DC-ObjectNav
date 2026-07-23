import heapq
import torch
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

_SQRT2 = float(np.sqrt(2.0))


def nearest_free_cell(occ_map, goal_zx, seed_zx):
    """Nearest observed-FREE cell (occ==1) to `goal_zx` reachable from `seed_zx`.

    Used by EXPLOIT-mode A* to snap the detector-projected goal — which usually
    lands on the target object's surface (an occupied cell) and may sit beyond
    the observed-free frontier — onto the closest cell the agent can actually
    stand on and reach *through observed-free space*. The candidate set is the
    8-connected FREE component containing `seed_zx` (the agent), so the returned
    cell is guaranteed connected to the agent by a free-cell path; as the agent
    advances and observes more floor, this creeps toward the object each replan.

    The seed itself may be occupied/unseen (agent dilated into a wall, or on an
    unmapped sliver) — then it attaches to the free component of the nearest
    free cell. Returns (z, x), or None if no free cell is reachable at all.
    """
    occ = np.asarray(occ_map)
    H, W = occ.shape
    free = occ == 1
    labels, _ = ndimage.label(free, structure=np.ones((3, 3), dtype=bool))
    sz = min(max(int(round(float(seed_zx[0]))), 0), H - 1)
    sx = min(max(int(round(float(seed_zx[1]))), 0), W - 1)
    seed_label = labels[sz, sx]
    if seed_label == 0:
        fz, fx = np.where(free)
        if fz.size == 0:
            return None
        j = int(np.argmin((fz - sz) ** 2 + (fx - sx) ** 2))
        seed_label = labels[fz[j], fx[j]]
    cz, cx = np.where(labels == seed_label)
    gz = min(max(int(round(float(goal_zx[0]))), 0), H - 1)
    gx = min(max(int(round(float(goal_zx[1]))), 0), W - 1)
    j = int(np.argmin((cz - gz) ** 2 + (cx - gx) ** 2))
    return (int(cz[j]), int(cx[j]))


def astar_free(occ_map, start_zx, goal_zx):
    """8-connected A* from `start_zx` to `goal_zx` through observed-FREE cells.

    Only cells with occ==1 are traversable — occupied (>=2) AND unseen (0) cells
    are impassable, so the path never routes through unexplored space. Diagonal
    steps cost √2. No anti-corner-cut restriction: `reachable_min`'s connectivity
    (used by the MPPI goal-distance field) already treats the grid as plain
    8-connected with no corner rule, and adding one here caused a confirmed
    real failure — a diagonal-adjacent single occupied cell disconnected a
    region `reachable_min`/plain 8-connectivity considered fine, permanently
    sealing an otherwise-open approach and wedging the agent. At this grid's
    resolution (~0.05-0.10m) a cell reading "occupied" is a coarse
    approximation of the true continuous geometry, not a guarantee of zero
    clearance at its corner, so disallowing the cut bought unrealistic caution
    at the cost of real deadlocks. The start cell is always allowed to be
    stepped out of (the agent may be standing on an occupied/unmapped cell).
    Returns the path as a list of (z, x) cells start→goal inclusive, or None if
    the goal is not reachable through free space.
    """
    occ = np.asarray(occ_map)
    H, W = occ.shape
    sz = min(max(int(round(float(start_zx[0]))), 0), H - 1)
    sx = min(max(int(round(float(start_zx[1]))), 0), W - 1)
    gz = min(max(int(round(float(goal_zx[0]))), 0), H - 1)
    gx = min(max(int(round(float(goal_zx[1]))), 0), W - 1)
    if (sz, sx) == (gz, gx):
        return [(sz, sx)]
    free = occ == 1
    if not free[gz, gx]:
        return None

    def heur(z, x):
        dz, dx = abs(z - gz), abs(x - gx)
        return (_SQRT2 - 1.0) * min(dz, dx) + max(dz, dx)  # octile

    neighbors = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2))
    open_heap = [(heur(sz, sx), 0.0, sz, sx)]
    g_cost = {(sz, sx): 0.0}
    came = {}
    while open_heap:
        _, gc, z, x = heapq.heappop(open_heap)
        if (z, x) == (gz, gx):
            path = [(z, x)]
            while (z, x) in came:
                z, x = came[(z, x)]
                path.append((z, x))
            path.reverse()
            return path
        if gc > g_cost.get((z, x), float('inf')):
            continue
        for dz, dx, step in neighbors:
            nz, nx = z + dz, x + dx
            if not (0 <= nz < H and 0 <= nx < W):
                continue
            if not free[nz, nx]:
                continue
            ng = gc + step
            if ng < g_cost.get((nz, nx), float('inf')):
                g_cost[(nz, nx)] = ng
                came[(nz, nx)] = (z, x)
                heapq.heappush(open_heap, (ng + heur(nz, nx), ng, nz, nx))
    return None


def reachable_min(field, occ_map, seed_zx):
    """Min of `field` over the observed-FREE cells of the 8-connected
    non-occupied component containing `seed_zx` — i.e. the best field value at
    a spot the agent can reach AND has actually observed to be standable.

    Used to anchor the goal distance field. Two hijack modes are guarded:
    * an enclosed observed-free or unseen pocket inside the goal object's blob
      (floor slivers seen under/behind furniture) — excluded by the component
      test (not 8-connected to the seed);
    * the unseen EXTERIOR of the scene for a goal on an outer wall (e.g. a
      window): unseen cells beyond the wall connect to the seed's component
      through frontier/map-edge cells and sit within a step or two of the goal,
      so an unrestricted min deflates the anchor to a cell on the far side of
      the wall — every interior cell then keeps the full wall-crossing penalty
      and the arrival radius (and the caller's stop check) becomes
      unsatisfiable, orbiting the agent forever. Restricting the min to
      observed-FREE members keeps the anchor on real floor near the goal.

    Connectivity still runs over all non-occupied cells (unseen doesn't
    disconnect the component). Falls back to the unrestricted component min
    when it holds no observed-free cell yet (degenerate early-episode maps).
    The seed cell is treated as traversable regardless of its occupancy (the
    agent may start dilated into a wall), which bridges it to its adjacent
    non-occupied cells.
    """
    occ = np.asarray(occ_map)
    sz = min(max(int(round(float(seed_zx[0]))), 0), occ.shape[0] - 1)
    sx = min(max(int(round(float(seed_zx[1]))), 0), occ.shape[1] - 1)
    mask = occ < 2
    mask[sz, sx] = True
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    comp = labels == labels[sz, sx]
    anchor_cells = comp & (occ == 1)
    if not anchor_cells.any():
        anchor_cells = comp
    return float(np.asarray(field)[anchor_cells].min())


def goal_distance_field(occ_map, goal_zx, occupied_cell_cost, unseen_cell_cost=1.0):
    """Obstacle-aware distance-to-go (in cells) from every BEV cell to `goal_zx`.

    Single-source Dijkstra over the 8-connected grid (diagonal steps cost √2),
    seeded at the goal cell. Entering a free cell costs the step length;
    entering an occupied cell costs step length × `occupied_cell_cost`, and
    entering an UNSEEN cell costs step length × `unseen_cell_cost`. Occupied
    cells are deliberately traversable-at-a-price rather than blocked: the
    goal is usually projected onto the target object's surface (an occupied
    blob), so the wavefront must be able to escape a few buried cells — while
    crossing a wall costs ~thickness × occupied_cell_cost, far more than any
    indoor detour, so the field routes around geometry instead of leaking
    through it. `unseen_cell_cost` > 1 makes the field prefer routes through
    observed-free space over optimistic shortcuts through unexplored cells
    (which may hide walls and force a backtrack); 1.0 treats unseen like free.

    Returns a float32 (H, W) array, finite everywhere (the penalized graph is
    fully connected).
    """
    occ = np.asarray(occ_map)
    H, W = occ.shape
    n = H * W
    cell_cost = np.where(occ >= 2, float(occupied_cell_cost),
                         np.where(occ == 0, float(unseen_cell_cost), 1.0)).ravel()

    rows, cols, weights = [], [], []
    # 4 unique neighbor offsets; both edge directions added explicitly since
    # the weight is the cost of the *entered* cell (directed graph).
    for dz, dx, step in ((0, 1, 1.0), (1, 0, 1.0),
                         (1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0))):
        z0, z1 = max(0, -dz), min(H, H - dz)
        x0, x1 = max(0, -dx), min(W, W - dx)
        zz, xx = np.meshgrid(np.arange(z0, z1), np.arange(x0, x1), indexing='ij')
        u = (zz * W + xx).ravel()
        v = ((zz + dz) * W + (xx + dx)).ravel()
        rows.append(u); cols.append(v); weights.append(step * cell_cost[v])
        rows.append(v); cols.append(u); weights.append(step * cell_cost[u])

    graph = csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n))
    src = int(goal_zx[0]) * W + int(goal_zx[1])
    dist = dijkstra(graph, directed=True, indices=src)
    return dist.reshape(H, W).astype(np.float32)


def compute_batch_fov_ig(Z_samples, X_samples, Theta_samples, epi_map, occ_map, cfg, device="cuda", fov_deg=90, max_dist=None, gamma_ig=0.95, intrinsics=None, sensor_height=1.5):
    if max_dist is None:
        max_dist = int(cfg.max_sensor_dist / cfg.voxel_resolution)
    """
    Compute discounted Information Gain (IG) along a batch of trajectories.
    Z_samples, X_samples, Theta_samples: [K, H] tensors.
    """
    K, H_length = Z_samples.shape
    if H_length < 1:
        return torch.zeros(K, device=device), torch.zeros((K, epi_map.shape[0], epi_map.shape[1]), dtype=torch.bool, device=device)

    Z_dim, X_dim = epi_map.shape

    seen_so_far = torch.zeros((K, Z_dim, X_dim), dtype=torch.bool, device=device)
    total_discounted_ig = torch.zeros(K, device=device)

    # 1. Setup Camera Metrics & Dead Zone
    if intrinsics is not None:
        fx, fy, cx, cy, H, W = intrinsics
        num_rays = int(W // 4)
        pixels = torch.linspace(0, W - 1, num_rays, device=device)
        ray_angles_relative = torch.atan((pixels - cx) / fx)

        vfov = 2 * np.arctan(H / (2 * fy))
        min_dist = sensor_height / np.tan(vfov / 2)
    else:
        fov_rad = np.radians(fov_deg)
        num_rays = int(fov_deg)
        ray_angles_relative = torch.linspace(-fov_rad/2, fov_rad/2, num_rays, device=device)
        min_dist = 1.0

    min_dist_cells = max(1.0, min_dist / cfg.voxel_resolution)
    all_steps = torch.arange(1, max_dist + 1, device=device).view(1, 1, -1)

    for t in range(H_length):
        pos_z = Z_samples[:, t] # [K]
        pos_x = X_samples[:, t] # [K]
        heading = Theta_samples[:, t] # [K]

        ray_angles = ray_angles_relative.view(1, -1) + heading.view(-1, 1) # [K, num_rays]
        ray_angles_vec = ray_angles.unsqueeze(-1) # [K, num_rays, 1]

        pos_z_vec = pos_z.view(-1, 1, 1) # [K, 1, 1]
        pos_x_vec = pos_x.view(-1, 1, 1) # [K, 1, 1]

        ray_z = (pos_z_vec + all_steps * torch.sin(ray_angles_vec)).long() # [K, num_rays, max_dist]
        ray_x = (pos_x_vec + all_steps * torch.cos(ray_angles_vec)).long() # [K, num_rays, max_dist]

        valid_idx = (ray_z >= 0) & (ray_z < Z_dim) & (ray_x >= 0) & (ray_x < X_dim)

        ray_z_clamp = torch.clamp(ray_z, 0, Z_dim - 1)
        ray_x_clamp = torch.clamp(ray_x, 0, X_dim - 1)

        occ_samples = occ_map[ray_z_clamp, ray_x_clamp]
        occ_samples[~valid_idx] = 0

        is_occ = (occ_samples >= 2)
        occ_cumsum = torch.cumsum(is_occ.int(), dim=2)
        shift_cumsum = torch.cat([torch.zeros((K, num_rays, 1), dtype=torch.int32, device=device), occ_cumsum[:, :, :-1]], dim=2)

        is_visible = (shift_cumsum == 0) & valid_idx & (all_steps >= min_dist_cells)

        waypoint_mask = torch.zeros((K, Z_dim, X_dim), dtype=torch.bool, device=device)
        k_idx = torch.arange(K, device=device).view(-1, 1, 1).expand(-1, num_rays, max_dist)

        vis_k = k_idx[is_visible]
        vis_z = ray_z[is_visible]
        vis_x = ray_x[is_visible]

        if vis_z.numel() > 0:
            waypoint_mask[vis_k, vis_z, vis_x] = True

        new_info_mask = waypoint_mask & (~seen_so_far)
        ig_t = (new_info_mask * epi_map.unsqueeze(0)).sum(dim=(1, 2))

        total_discounted_ig += (gamma_ig ** t) * ig_t
        seen_so_far |= waypoint_mask

    return total_discounted_ig, seen_so_far
