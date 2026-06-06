import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import heapq
import random
import math
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Warehouse Robot AI",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Warehouse Robot — Intelligent AI Navigation System")
st.markdown(
    "Simulates a warehouse robot using **A\\* Search**, **Genetic Algorithm**, "
    "and **K-Means Clustering** to pick up shelf items in an optimal order."
)

# ─────────────────────────────────────────────
#  Sidebar – configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)
    num_items = st.slider("Number of Items to Pick", min_value=2, max_value=16, value=8)

    st.markdown("---")
    st.subheader("Genetic Algorithm")
    pop_size = st.slider("Population Size", 20, 200, 50, step=10)
    generations = st.slider("Generations", 25, 300, 100, step=25)
    mutation_rate = st.slider("Mutation Rate", 0.05, 0.5, 0.25, step=0.05)

    st.markdown("---")
    st.subheader("K-Means Clustering")
    n_zones = st.slider("Number of Zones", 2, 5, 3)

    run = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
#  Core algorithms  (identical to notebook)
# ─────────────────────────────────────────────

def build_warehouse(seed_val, n_items):
    random.seed(seed_val)
    np.random.seed(seed_val)

    GRID_SIZE = 15
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    rack_cols = [2, 5, 8, 11]
    rack_rows = [(1, 5), (7, 11), (1, 5), (7, 11)]
    for col, (r_start, r_end) in zip(rack_cols, rack_rows):
        for r in range(r_start, r_end + 1):
            grid[r][col] = 1

    START = (13, 0)
    SHELVES = [
        (1, 1),  (1, 4),  (1, 7),  (1, 10), (1, 13),
        (4, 1),  (4, 4),  (4, 7),  (4, 10), (4, 13),
        (8, 1),  (8, 4),  (8, 7),  (8, 10), (8, 13),
        (11, 1), (11, 4), (11, 7), (11, 10), (11, 13),
    ]
    n_items = min(n_items, len(SHELVES))
    ORDER_ITEMS = random.sample(SHELVES, n_items)

    return grid, START, SHELVES, ORDER_ITEMS, GRID_SIZE


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    g_scores = {start: 0}
    nodes_explored = 0

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            return path, g, nodes_explored

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                new_g = g + 1
                if neighbor not in g_scores or new_g < g_scores[neighbor]:
                    g_scores[neighbor] = new_g
                    heapq.heappush(open_set, (
                        new_g + heuristic(neighbor, goal),
                        new_g, neighbor, path + [neighbor]
                    ))

    return None, float('inf'), nodes_explored


def get_full_route_astar(grid, start, item_order):
    full_path, total_cost, all_nodes = [], 0, 0
    current = start
    for item in item_order:
        path, cost, nodes = astar(grid, current, item)
        if path:
            full_path.extend(path[1:])
            total_cost += cost
            all_nodes += nodes
            current = item
    path, cost, nodes = astar(grid, current, start)
    if path:
        full_path.extend(path[1:])
        total_cost += cost
        all_nodes += nodes
    return full_path, total_cost, all_nodes


def build_distance_cache(grid, start, items):
    cache = {}
    all_points = [start] + items
    for i in range(len(all_points)):
        for j in range(i + 1, len(all_points)):
            a, b = all_points[i], all_points[j]
            key = (a, b)
            if key not in cache:
                _, cost, _ = astar(grid, a, b)
                cache[(a, b)] = cost
                cache[(b, a)] = cost
    return cache


def route_cost_cached(cache, start, order):
    total = cache.get((start, order[0]), float('inf'))
    for i in range(len(order) - 1):
        total += cache.get((order[i], order[i + 1]), float('inf'))
    total += cache.get((order[-1], start), float('inf'))
    return total


def fitness_fn(cache, start, chromosome, items):
    order = [items[i] for i in chromosome]
    cost = route_cost_cached(cache, start, order)
    return 1.0 / cost if cost > 0 else float('inf')


def tournament_selection(population, fitnesses, k=3):
    candidates = random.sample(range(len(population)), k)
    return max(candidates, key=lambda i: fitnesses[i])


def ordered_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b + 1] = p1[a:b + 1]
    remaining = [x for x in p2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[idx]
            idx += 1
    return child


def swap_mutation(chromosome, rate=0.2):
    chrom = chromosome[:]
    if random.random() < rate:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def genetic_algorithm(cache, start, items, pop_size=50, generations=100,
                      mutation_rate=0.25, elitism=2):
    n = len(items)
    population = [random.sample(range(n), n) for _ in range(pop_size)]
    best_cost_history = []
    best_chrom, best_cost = None, float('inf')

    for gen in range(generations):
        fitnesses = [fitness_fn(cache, start, chrom, items) for chrom in population]

        for chrom, fit in zip(population, fitnesses):
            cost = 1.0 / fit if fit > 0 else float('inf')
            if cost < best_cost:
                best_cost = cost
                best_chrom = chrom[:]

        best_cost_history.append(best_cost)

        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        new_pop = [ind for _, ind in sorted_pop[:elitism]]

        while len(new_pop) < pop_size:
            p1 = population[tournament_selection(population, fitnesses)]
            p2 = population[tournament_selection(population, fitnesses)]
            child = swap_mutation(ordered_crossover(p1, p2), mutation_rate)
            new_pop.append(child)

        population = new_pop

    return [items[i] for i in best_chrom], best_cost, best_cost_history


def kmeans_zone_order(items, start, n_clusters=3):
    coords = np.array(items)
    n_clusters = min(n_clusters, len(items))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    start_arr = np.array(start)
    cluster_order = np.argsort([np.linalg.norm(centers[k] - start_arr) for k in range(n_clusters)])
    zone_order = []
    for cid in cluster_order:
        cluster_items = [items[i] for i in range(len(items)) if labels[i] == cid]
        cluster_items.sort(key=lambda p: math.dist(p, tuple(centers[cid])))
        zone_order.extend(cluster_items)
    return labels, centers, zone_order


# ─────────────────────────────────────────────
#  Visualizations
# ─────────────────────────────────────────────

CLUSTER_COLORS = ['#E53935', '#43A047', '#1E88E5', '#FB8C00', '#8E24AA']


def draw_warehouse(grid, start, shelves, order_items, path=None,
                   title='Warehouse Layout', cluster_labels=None, GRID_SIZE=15):
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = ListedColormap(['#F5F5F5', '#607D8B'])
    ax.imshow(grid.copy().astype(float), cmap=cmap, origin='upper')

    for x in range(grid.shape[1] + 1):
        ax.axvline(x - 0.5, color='#BDBDBD', linewidth=0.5)
    for y in range(grid.shape[0] + 1):
        ax.axhline(y - 0.5, color='#BDBDBD', linewidth=0.5)

    for (r, c) in shelves:
        color = '#90CAF9'
        if cluster_labels is not None and (r, c) in order_items:
            idx = order_items.index((r, c))
            color = CLUSTER_COLORS[cluster_labels[idx] % len(CLUSTER_COLORS)]
        elif (r, c) in order_items:
            color = '#FFB300'

        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.35, r - 0.35), 0.70, 0.70,
            boxstyle='round,pad=0.05', color=color, zorder=3))
        ax.text(c, r, 'S', ha='center', va='center', fontsize=7,
                fontweight='bold', color='white', zorder=4)

    if path:
        ax.plot([p[1] for p in path], [p[0] for p in path],
                color='#E91E63', linewidth=2.5, linestyle='--', zorder=5, alpha=0.8)

    ax.add_patch(plt.Circle((start[1], start[0]), 0.4, color='#00BCD4', zorder=6))
    ax.text(start[1], start[0], 'R', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=7)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE), fontsize=7)
    ax.set_yticklabels(range(GRID_SIZE), fontsize=7)

    legend_handles = [
        mpatches.Patch(color='#90CAF9', label='Shelf (not in order)'),
        mpatches.Patch(color='#FFB300', label='Item to Pick'),
        mpatches.Patch(color='#607D8B', label='Obstacle / Rack'),
        mpatches.Patch(color='#00BCD4', label='Robot Base'),
    ]
    if path:
        legend_handles.append(
            plt.Line2D([0], [0], color='#E91E63', linestyle='--', linewidth=2, label='Robot Path'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.9)
    plt.tight_layout()
    return fig


def draw_ga_convergence(ga_history, naive_cost, ga_cost):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(ga_history, color='#E91E63', linewidth=2, label='GA Best Cost per Generation')
    ax.axhline(naive_cost, color='#607D8B', linestyle='--', linewidth=1.5,
               label=f'Naive Order ({naive_cost:.0f} steps)')
    ax.axhline(ga_cost, color='#43A047', linestyle='--', linewidth=1.5,
               label=f'GA Best ({ga_cost:.0f} steps)')
    ax.fill_between(range(len(ga_history)), ga_history, alpha=0.12, color='#E91E63')
    ax.set_title('GA Convergence Over Generations', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Total Route Cost (steps)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def draw_comparison(strategies, costs):
    naive_cost = costs[0]
    bar_colors = ['#607D8B', '#1E88E5', '#43A047']
    improvements = [(naive_cost - c) / naive_cost * 100 for c in costs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    bars = axes[0].bar(strategies, costs, color=bar_colors, edgecolor='white', width=0.5)
    for bar, cost in zip(bars, costs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{cost:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_title('Total Route Cost by Strategy', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Total Steps')
    axes[0].set_ylim(0, max(costs) * 1.25)
    axes[0].grid(axis='y', alpha=0.3)

    bars2 = axes[1].bar(strategies, improvements, color=bar_colors, edgecolor='white', width=0.5)
    for bar, imp in zip(bars2, improvements):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{imp:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].set_title('Improvement Over Naive Order', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Improvement (%)')
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('Algorithm Performance Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Main simulation
# ─────────────────────────────────────────────

if run:
    grid, START, SHELVES, ORDER_ITEMS, GRID_SIZE = build_warehouse(int(seed), num_items)

    # ── Initial Layout ──────────────────────────────────────
    st.subheader("📦 Warehouse Layout")
    fig_layout = draw_warehouse(grid, START, SHELVES, ORDER_ITEMS,
                                title='Initial Warehouse Layout', GRID_SIZE=GRID_SIZE)
    st.pyplot(fig_layout)
    plt.close(fig_layout)

    st.markdown(
        f"**Grid:** {GRID_SIZE}×{GRID_SIZE} &nbsp;|&nbsp; "
        f"**Robot start:** {START} &nbsp;|&nbsp; "
        f"**Items to pick:** {len(ORDER_ITEMS)}"
    )
    with st.expander("Item locations"):
        for i, loc in enumerate(ORDER_ITEMS, 1):
            st.write(f"{i}. {loc}")

    st.divider()

    # ── A* demo ────────────────────────────────────────────
    st.subheader("🔍 Algorithm 1: A* Pathfinding")
    with st.spinner("Running A* single-path test…"):
        test_path, test_cost, test_nodes = astar(grid, START, ORDER_ITEMS[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Path Length", f"{test_cost} steps")
    col2.metric("Nodes Explored", test_nodes)
    col3.metric("Destination", str(ORDER_ITEMS[0]))

    st.divider()

    # ── Distance cache ─────────────────────────────────────
    with st.spinner("Pre-computing A* distance cache…"):
        dist_cache = build_distance_cache(grid, START, ORDER_ITEMS)

    naive_cost = route_cost_cached(dist_cache, START, ORDER_ITEMS)

    # ── Genetic Algorithm ──────────────────────────────────
    st.subheader("🧬 Algorithm 2: Genetic Algorithm")
    prog = st.progress(0, text="Evolving pickup order…")

    # Run GA with progress updates
    random.seed(int(seed))
    n = len(ORDER_ITEMS)
    population = [random.sample(range(n), n) for _ in range(pop_size)]
    best_cost_history = []
    best_chrom, best_cost_ga = None, float('inf')

    for gen in range(generations):
        fitnesses = [fitness_fn(dist_cache, START, chrom, ORDER_ITEMS) for chrom in population]
        for chrom, fit in zip(population, fitnesses):
            cost = 1.0 / fit if fit > 0 else float('inf')
            if cost < best_cost_ga:
                best_cost_ga = cost
                best_chrom = chrom[:]
        best_cost_history.append(best_cost_ga)
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        new_pop = [ind for _, ind in sorted_pop[:2]]
        while len(new_pop) < pop_size:
            p1 = population[tournament_selection(population, fitnesses)]
            p2 = population[tournament_selection(population, fitnesses)]
            child = swap_mutation(ordered_crossover(p1, p2), mutation_rate)
            new_pop.append(child)
        population = new_pop
        prog.progress((gen + 1) / generations, text=f"Generation {gen + 1}/{generations}")

    ga_order = [ORDER_ITEMS[i] for i in best_chrom]
    prog.empty()

    col1, col2, col3 = st.columns(3)
    col1.metric("Naive Order Cost", f"{naive_cost:.0f} steps")
    col2.metric("GA Optimized Cost", f"{best_cost_ga:.0f} steps")
    improvement = (naive_cost - best_cost_ga) / naive_cost * 100
    col3.metric("Improvement", f"{improvement:.1f}%", delta=f"-{naive_cost - best_cost_ga:.0f} steps")

    fig_ga = draw_ga_convergence(best_cost_history, naive_cost, best_cost_ga)
    st.pyplot(fig_ga)
    plt.close(fig_ga)

    st.divider()

    # ── K-Means ────────────────────────────────────────────
    st.subheader("📍 Algorithm 3: K-Means Clustering")
    km_labels, km_centers, km_order = kmeans_zone_order(ORDER_ITEMS, START, n_clusters=n_zones)
    km_cost = route_cost_cached(dist_cache, START, km_order)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("K-Means Cost", f"{km_cost:.0f} steps")
        for z in range(n_zones):
            zone_items = [ORDER_ITEMS[i] for i in range(len(ORDER_ITEMS)) if km_labels[i] == z]
            st.write(f"**Zone {z + 1}:** {zone_items}")

    with col2:
        fig_km = draw_warehouse(grid, START, SHELVES, ORDER_ITEMS,
                                title=f'K-Means — {n_zones} Zones',
                                cluster_labels=list(km_labels), GRID_SIZE=GRID_SIZE)
        st.pyplot(fig_km)
        plt.close(fig_km)

    st.divider()

    # ── Full Simulation ────────────────────────────────────
    st.subheader("🚀 Full Robot Simulation (GA + A*)")
    with st.spinner("Running full route simulation…"):
        full_path, total_cost, total_nodes = get_full_route_astar(grid, START, ga_order)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Steps", total_cost)
    col2.metric("Nodes Explored", total_nodes)
    col3.metric("Items Picked", len(ga_order))

    fig_sim = draw_warehouse(
        grid, START, SHELVES, ORDER_ITEMS,
        path=[START] + full_path,
        title='GA-Optimised Route with A* Pathfinding',
        GRID_SIZE=GRID_SIZE
    )
    st.pyplot(fig_sim)
    plt.close(fig_sim)

    with st.expander("Pickup Sequence"):
        for i, item in enumerate(ga_order, 1):
            st.write(f"{i}. Shelf at {item}")
        st.write(f"{len(ga_order) + 1}. Return to Base {START}")

    st.divider()

    # ── Performance Comparison ─────────────────────────────
    st.subheader("📊 Performance Comparison")
    strategies = ['Naive Order', 'K-Means Zones', 'GA Optimized']
    costs = [naive_cost, km_cost, best_cost_ga]

    fig_cmp = draw_comparison(strategies, costs)
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    st.markdown("#### Summary Table")
    improvements_all = [(naive_cost - c) / naive_cost * 100 for c in costs]
    for s, c, imp in zip(strategies, costs, improvements_all):
        st.write(f"**{s}** — {c:.0f} steps &nbsp;|&nbsp; {imp:.1f}% improvement over naive")

else:
    st.info("Configure the parameters in the sidebar and click **▶ Run Simulation** to start.")
    st.markdown("""
    ### How it works
    | Algorithm | Role |
    |-----------|------|
    | **A\\* Search** | Finds the shortest obstacle-free path between two grid cells |
    | **Genetic Algorithm** | Optimises the item pickup sequence (TSP variant) |
    | **K-Means Clustering** | Groups shelves into geographic zones for structured traversal |
    """)
