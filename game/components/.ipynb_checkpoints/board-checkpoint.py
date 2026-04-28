## This will define how the board behaves -> one big graph structure
import math
import random
import numpy as np
import networkx as nx
import opensimplex

from game.enums import UnitType, TileType, BoardType, PlayerId, TileStatus, NODE_FEAT_DIM, N_PLAYERS
from game.components.tile import Tile
from game.components.city import City
from game.components.units import Warrior

# ─── constants ────────────────────────────────────────────────────────────────
MOUNTAIN_DENSITY_DRYLANDS    = 0.15
MOUNTAIN_DENSITY_LAKES       = 0.10
MOUNTAIN_DENSITY_ARCHIPELAGO = 0.10
LAKES_WATER_FRACTION         = 0.275   # target midpoint of 25–30 %


# ─── distance helpers ─────────────────────────────────────────────────────────

def _chebyshev(r1, c1, r2, c2):
    return max(abs(r1 - r2), abs(c1 - c2))


def _edge_distance_matrix(H, W):
    """Min Chebyshev distance from (i,j) to any board edge."""
    rows = np.minimum(np.arange(H), H - 1 - np.arange(H))[:, None]
    cols = np.minimum(np.arange(W), W - 1 - np.arange(W))[None, :]
    return np.minimum(rows, cols)


def _min_dist_to_set(H, W, positions):
    """(H, W) matrix of min Chebyshev distance from each cell to any position in *positions*."""
    if not positions:
        return np.full((H, W), 999, dtype=int)
    rs = np.array([p[0] for p in positions])
    cs = np.array([p[1] for p in positions])
    row_idx = np.arange(H)[:, None, None]
    col_idx = np.arange(W)[None, :, None]
    dist = np.maximum(np.abs(row_idx - rs), np.abs(col_idx - cs))  # (H, W, N)
    return dist.min(axis=2)


def _chebyshev_to(H, W, r, c):
    """(H, W) matrix of Chebyshev distances from every cell to (r, c)."""
    return np.maximum(
        np.abs(np.arange(H)[:, None] - r),
        np.abs(np.arange(W)[None, :] - c),
    )


# ─── capital placement ────────────────────────────────────────────────────────

def _place_capitals(H, W, n_players, edge_dist):
    """Place one capital per player in randomly selected quadrants.

    Constraints: edge_dist >= 2, pairwise Chebyshev distance >= 3.
    Retries up to 200 times if constraints cannot be met.
    """
    half_r, half_c = H // 2, W // 2
    quadrants = [
        (0,      half_r, 0,      half_c),
        (0,      half_r, half_c, W),
        (half_r, H,      0,      half_c),
        (half_r, H,      half_c, W),
    ]

    for _ in range(200):
        chosen_quads = random.sample(quadrants, n_players)
        caps = []
        ok = True
        for (r0, r1, c0, c1) in chosen_quads:
            eligible = [
                (r, c)
                for r in range(r0, r1)
                for c in range(c0, c1)
                if edge_dist[r, c] >= 2
                and all(_chebyshev(r, c, pr, pc) >= 3 for (pr, pc) in caps)
            ]
            if not eligible:
                ok = False
                break
            caps.append(random.choice(eligible))
        if ok:
            return caps

    # Fallback: pick anywhere with edge_dist >= 2
    eligible = list(zip(*np.where(edge_dist >= 2)))
    return [tuple(int(x) for x in eligible[i])
            for i in random.sample(range(len(eligible)), n_players)]


# ─── village placement ────────────────────────────────────────────────────────

def _place_suburb_villages(H, W, capitals, all_cities, edge_dist):
    """Place up to 2 suburb villages per capital.

    Constraints: Chebyshev distance 1–3 from parent capital,
    edge_dist >= 2, Chebyshev distance >= 3 from every other city.
    Villages are added one at a time so they constrain each other.
    """
    suburbs = []
    current = list(all_cities)
    for (cr, cc) in capitals:
        cap_dist = _chebyshev_to(H, W, cr, cc)
        for _ in range(2):
            min_to_cur = _min_dist_to_set(H, W, current)
            eligible = np.argwhere(
                (cap_dist >= 1) & (cap_dist <= 3) &
                (edge_dist >= 2) &
                (min_to_cur >= 3)
            )
            if len(eligible) == 0:
                break
            pos = tuple(eligible[random.randint(0, len(eligible) - 1)])
            pos = (int(pos[0]), int(pos[1]))
            suburbs.append(pos)
            current.append(pos)
    return suburbs


def _n_preterrain_villages(W, n_capitals, n_suburbs):
    return max(0, math.ceil(((W / 3) ** 2 - (n_capitals + n_suburbs)) * 0.3))


def _place_villages_greedily(H, W, n, existing_cities, edge_min, edge_dist):
    """Place up to n villages one at a time.

    Constraints: edge_dist >= edge_min, Chebyshev distance >= 3 from all placed cities.
    """
    placed = []
    current = list(existing_cities)
    for _ in range(n):
        min_to_cur = _min_dist_to_set(H, W, current) if current else np.full((H, W), 999, dtype=int)
        eligible = np.argwhere((edge_dist >= edge_min) & (min_to_cur >= 3))
        if len(eligible) == 0:
            break
        pos = eligible[random.randint(0, len(eligible) - 1)]
        pos = (int(pos[0]), int(pos[1]))
        placed.append(pos)
        current.append(pos)
    return placed


def _place_postterrain_villages(H, W, tile_type, existing_cities, edge_dist):
    """Greedily fill all post-terrain eligible positions.

    Constraints: field tile, edge_dist >= 4 (3 tiles between village and edge),
    Chebyshev distance >= 3 from all placed cities (2 tiles in between).
    """
    placed = []
    current = list(existing_cities)
    while True:
        min_to_cur = _min_dist_to_set(H, W, current) if current else np.full((H, W), 999, dtype=int)
        eligible = np.argwhere(
            (tile_type == TileType.field) &
            (edge_dist >= 4) &
            (min_to_cur >= 3)
        )
        if len(eligible) == 0:
            break
        pos = eligible[random.randint(0, len(eligible) - 1)]
        pos = (int(pos[0]), int(pos[1]))
        placed.append(pos)
        current.append(pos)
    return placed


# ─── terrain generation ───────────────────────────────────────────────────────

def _terrain_drylands(H, W, protected_set):
    """Fields + mountains (~15%), no water."""
    protected_mask = np.zeros((H, W), dtype=bool)
    for (r, c) in protected_set:
        protected_mask[r, c] = True
    tile_type = np.zeros((H, W), dtype=int)
    candidates = ~protected_mask
    mountain_mask = candidates & (np.random.rand(H, W) < MOUNTAIN_DENSITY_DRYLANDS)
    tile_type[mountain_mask] = TileType.mountain
    return tile_type


def _any_capital_isolated(tile_type, capitals, H, W):
    """True if any capital has all accessible 8-neighbours as water."""
    for (cr, cc) in capitals:
        surrounded = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and tile_type[nr, nc] != TileType.water:
                    surrounded = False
                    break
            if not surrounded:
                break
        if surrounded:
            return True
    return False


def _terrain_lakes(H, W, protected_set, capitals, edge_dist, max_retries=30):
    """25–30% water via simplex noise; edge tiles and protected tiles stay field.
    Restarts if any capital ends up completely surrounded by water.
    """
    protected_mask = np.zeros((H, W), dtype=bool)
    for (r, c) in protected_set:
        protected_mask[r, c] = True

    for _ in range(max_retries):
        seed = random.randint(0, 100_000)
        opensimplex.seed(seed)
        offset_r = random.uniform(0, 100)
        offset_c = random.uniform(0, 100)
        scale = random.uniform(0.15, 0.35)

        noise_field = np.array([
            [opensimplex.noise2(offset_r + r * scale, offset_c + c * scale) for c in range(W)]
            for r in range(H)
        ])

        # Tiles eligible to become water: interior (edge_dist > 0), not protected
        candidate_mask = (edge_dist > 0) & ~protected_mask
        n_candidates = int(candidate_mask.sum())
        if n_candidates == 0:
            continue

        # Adjust fraction so total water / (H*W) hits the target
        target_water_tiles = LAKES_WATER_FRACTION * H * W
        candidate_fraction = min(1.0, target_water_tiles / n_candidates)
        threshold = np.percentile(noise_field[candidate_mask], (1.0 - candidate_fraction) * 100)

        tile_type = np.zeros((H, W), dtype=int)
        tile_type[(noise_field >= threshold) & candidate_mask] = TileType.water

        if _any_capital_isolated(tile_type, capitals, H, W):
            continue  # restart

        # Mountains on interior field tiles that aren't protected
        mountain_cands = np.argwhere((tile_type == TileType.field) & ~protected_mask & (edge_dist > 0))
        n_mountains = int(len(mountain_cands) * MOUNTAIN_DENSITY_LAKES)
        if n_mountains > 0:
            idx = np.random.choice(len(mountain_cands), n_mountains, replace=False)
            tile_type[mountain_cands[idx, 0], mountain_cands[idx, 1]] = TileType.mountain

        return tile_type

    # Fallback: return last tile_type even if capitals are isolated
    return tile_type


def _terrain_archipelago(H, W, protected_set):
    """Water via random initialisation + optional cellular automata smoothing.
    Mountains placed on remaining field tiles.
    """
    protected_mask = np.zeros((H, W), dtype=bool)
    for (r, c) in protected_set:
        protected_mask[r, c] = True

    init_prob = random.uniform(0.4, 0.5)
    tile_type = (np.random.rand(H, W) < init_prob).astype(int)  # 1=water, 0=field
    tile_type[protected_mask] = 0

    # Smoothing iterations (currently 0; increase to add CA clustering)
    for _ in range(0):
        water = (tile_type == 1).astype(int)
        padded = np.pad(water, 1, mode='constant', constant_values=0)
        neighbours = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2]                     + padded[1:-1, 2:] +
            padded[2:,  :-2] + padded[2:,  1:-1]  + padded[2:,  2:]
        )
        tile_type = np.where(neighbours >= 5, 1, np.where(neighbours <= 2, 0, tile_type))
        tile_type[protected_mask] = 0

    # Mountains on remaining field tiles
    mountain_cands = np.argwhere((tile_type == TileType.field) & ~protected_mask)
    n_mountains = int(len(mountain_cands) * MOUNTAIN_DENSITY_ARCHIPELAGO)
    if n_mountains > 0:
        idx = np.random.choice(len(mountain_cands), n_mountains, replace=False)
        tile_type[mountain_cands[idx, 0], mountain_cands[idx, 1]] = TileType.mountain

    return tile_type


# ─── main entry point ─────────────────────────────────────────────────────────

def board_generating_logic(board_size, board_type, n_players):
    """Build the board plan matrix of shape (3, H, W):
      layer 0 – TileType int values
      layer 1 – village mask (1 = village or capital)
      layer 2 – capital mask (1 = capital)
    """
    H, W = board_size

    if board_type == BoardType.Dummy:
        water_matrix   = (np.random.rand(H, W) < 0.1).astype(int)
        village_matrix = (np.random.rand(H, W) < 0.1).astype(int) * (1 - water_matrix)
        chosen_capitals = random.sample(list(zip(*np.where(village_matrix == 1))), len(PlayerId))
        capital_matrix = np.zeros_like(village_matrix)
        for (r, c) in chosen_capitals:
            capital_matrix[r, c] = 1
        return np.stack([water_matrix, village_matrix, capital_matrix], axis=0)

    edge_dist = _edge_distance_matrix(H, W)
    tile_type  = np.zeros((H, W), dtype=int)
    village    = np.zeros((H, W), dtype=int)
    capital    = np.zeros((H, W), dtype=int)

    # 1. Capitals
    caps       = _place_capitals(H, W, n_players, edge_dist)
    all_cities = list(caps)

    # 2. Suburb villages (edge_dist >= 2, spacing >= 3)
    suburbs    = _place_suburb_villages(H, W, caps, all_cities, edge_dist)
    all_cities.extend(suburbs)

    # 3. Pre-terrain villages (edge_dist >= 2, spacing >= 3)
    n_pre   = _n_preterrain_villages(W, len(caps), len(suburbs))
    pre_vil = _place_villages_greedily(H, W, n_pre, all_cities, 2, edge_dist)
    all_cities.extend(pre_vil)

    # 4. Terrain
    protected_set = set(all_cities)
    if board_type == BoardType.Drylands:
        tile_type = _terrain_drylands(H, W, protected_set)
    elif board_type == BoardType.Lakes:
        tile_type = _terrain_lakes(H, W, protected_set, caps, edge_dist)
    elif board_type == BoardType.Archipelago:
        tile_type = _terrain_archipelago(H, W, protected_set)

    # 5. Post-terrain villages (edge_dist >= 4, spacing >= 3, field only)
    post_vil   = _place_postterrain_villages(H, W, tile_type, all_cities, edge_dist)
    all_cities.extend(post_vil)

    # 6. Assemble output matrices
    for (r, c) in suburbs + pre_vil + post_vil:
        village[r, c] = 1
    for (r, c) in caps:
        village[r, c] = 1
        capital[r, c] = 1

    return np.stack([tile_type, village, capital], axis=0)


class Board(object):
    """
    The board is holding all necessary objects to represent the full, visible game board.
    Individual board views will be handled in the player.py file.
    """
    def __init__(self, board_size, board_type, n_players):
        self.board_size = board_size
        self.board_type = board_type
        self.n_players = n_players

        self.board = []
        self.board_graph = np.empty(shape=(board_size[0] * board_size[1], NODE_FEAT_DIM))

        self.movement_topology_graph = nx.grid_2d_graph(board_size[0], board_size[1])
        ## add diagonal edges:
        diagonals = []
        for x in range(board_size[0] - 1):
            for y in range(board_size[1] - 1):
                diagonals.append(((x, y), (x + 1, y + 1)))
                diagonals.append(((x + 1, y), (x, y + 1)))
        self.movement_topology_graph.add_edges_from(diagonals)
        nx.set_edge_attributes(self.movement_topology_graph, 1.0, 'weight')

        self.initialize()

    def initialize(self, game=None):
        """This function creates an empty board based on the creation logic.
        It does not create the graph yet, the graph is steadily created from the board from the enum objects in a one-hot-encoded way.
        game is passed to allow unique unit ID generation via game._new_unit_id()."""
        board_matrix = board_generating_logic(self.board_size, self.board_type, self.n_players)
        capital_assign_counter = 0

        self.board = []
        self.capital_tile_ids = {}
        self.int_to_tup = {}
        self.tup_to_int = {}

        for ind, (i, j) in enumerate(np.ndindex(board_matrix.shape[1:])):
            design_vec = board_matrix[:, i, j]  # [tile_type, village, capital]

            self.int_to_tup[ind] = (i, j)
            self.tup_to_int[(i, j)] = ind

            city = None
            unit = None
            tile_status = TileStatus.no_status
            player_control = None

            field_type = TileType(design_vec[0])

            if (design_vec[1] and not design_vec[2]):
                city = City(player_id=None, tile_id=ind, is_capital=False)
            elif (design_vec[1] and design_vec[2]):
                city = City(player_id=PlayerId(capital_assign_counter), tile_id=ind, is_capital=True)
                self.capital_tile_ids[PlayerId(capital_assign_counter).value] = ind
                capital_assign_counter += 1

            tile = Tile(
                id=ind,
                tile_type=field_type,
                city=city,
                tile_status=tile_status,
                unit=unit,
                player_controls=player_control,
            )
            if city is not None:
                tile.has_road = True

            self.board.append(tile)

        self._update_road_edge_weights()

        ## place starting units:
        for player_id, capital_id in self.capital_tile_ids.items():
            uid = game._new_unit_id() if game is not None else player_id
            unit = Warrior(
                player_id=PlayerId(player_id),
                city=self.board[capital_id].city,
                tile=self.board[capital_id],
                unit_id=uid,
            )
            self.board[capital_id].unit = unit
            self.board[capital_id].city.current_n_units = 1

    def _update_road_edge_weights(self):
        """Set edge weight to 0.5 where both endpoints have has_road, else 1.0."""
        for u, v in self.movement_topology_graph.edges():
            u_id = self.tup_to_int[u]
            v_id = self.tup_to_int[v]
            w = 0.5 if (self.board[u_id].has_road and self.board[v_id].has_road) else 1.0
            self.movement_topology_graph[u][v]['weight'] = w

    def create_board_graph_from_board_state(self, active_tile_inds):
        """
        Uses self.board to create a one-hot encoded graph based on the current board state.
        TODO: A matrix mask would potentially be faster for uncovered_tile_ids...
        """
        for tile in self.board:
            if tile.id in active_tile_inds:
                self.board_graph[tile.id] = tile.transform_to_node_features()
