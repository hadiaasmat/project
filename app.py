
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import heapq
import random
import math
from sklearn.cluster import KMeans

random.seed(42)
np.random.seed(42)

st.set_page_config(page_title="Warehouse Robot Optimizer", page_icon="🤖", layout="wide")

GRID_SIZE = 15
warehouse_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

rack_cols = [2, 5, 8, 11]
rack_rows = [(1, 5), (7, 11), (1, 5), (7, 11)]

for col, (r_start, r_end) in zip(rack_cols, rack_rows):
    for r in range(r_start, r_end + 1):
        warehouse_grid[r][col] = 1

START = (13, 0)

SHELVES = [
    (1, 1), (1, 4), (1, 7), (1, 10), (1, 13),
    (4, 1), (4, 4), (4, 7), (4, 10), (4, 13),
    (8, 1), (8, 4), (8, 7), (8, 10), (8, 13),
    (11, 1), (11, 4), (11, 7), (11, 10), (11, 13),
]

distance_cache = {}

def visualize_warehouse(grid, start, shelves, order_items, path=None, title="Warehouse Layout", cluster_labels=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = ListedColormap(['#F5F5F5', '#607D8B'])
    ax.imshow(grid.copy().astype(float), cmap=cmap, origin='upper')

    for x in range(grid.shape[1] + 1):
        ax.axvline(x - 0.5, color='#BDBDBD', linewidth=0.5)
    for y in range(grid.shape[0] + 1):
        ax.axhline(y - 0.5, color='#BDBDBD', linewidth=0.5)

    cluster_colors = ['#E53935', '#43A047', '#1E88E5', '#FB8C00', '#8E24AA']

    for (r, c) in shelves:
        color = '#90CAF9'
        if cluster_labels is not None and (r, c) in order_items:
            idx = order_items.index((r, c))
            color = cluster_colors[cluster_labels[idx] % len(cluster_colors)]
        elif (r, c) in order_items:
            color = '#FFB300'

        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.35, r - 0.35), 0.70, 0.70,
            boxstyle='round,pad=0.05', color=color, zorder=3))

        ax.text(c, r, 'S', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    if path:
        ax.plot([p[1] for p in path], [p[0] for p in path],
                linewidth=2.5, linestyle='--')

    ax.add_patch(plt.Circle((start[1], start[0]), 0.4, color='#00BCD4'))
    ax.text(start[1], start[0], 'R', ha='center', va='center', color='white')

    ax.set_title(title)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    plt.tight_layout()
    return fig

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    g_scores = {start: 0}
    explored = 0

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        explored += 1

        if current == goal:
            return path, g, explored

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = current[0]+dr, current[1]+dc

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                neighbor = (nr, nc)
                ng = g + 1

                if neighbor not in g_scores or ng < g_scores[neighbor]:
                    g_scores[neighbor] = ng
                    heapq.heappush(open_set,
                                   (ng + heuristic(neighbor, goal),
                                    ng,
                                    neighbor,
                                    path + [neighbor]))
    return None, float('inf'), explored

def get_full_route_astar(grid, start, item_order):
    full_path = []
    total_cost = 0
    total_nodes = 0
    current = start

    for item in item_order:
        path, cost, nodes = astar(grid, current, item)
        if path:
            full_path.extend(path[1:])
            total_cost += cost
            total_nodes += nodes
            current = item

    path, cost, nodes = astar(grid, current, start)
    if path:
        full_path.extend(path[1:])
        total_cost += cost
        total_nodes += nodes

    return full_path, total_cost, total_nodes

def get_distance(grid, a, b):
    key = (a, b)
    if key not in distance_cache:
        _, cost, _ = astar(grid, a, b)
        distance_cache[key] = cost
        distance_cache[(b, a)] = cost
    return distance_cache[key]

def route_cost(grid, start, order):
    total = get_distance(grid, start, order[0])
    for i in range(len(order)-1):
        total += get_distance(grid, order[i], order[i+1])
    total += get_distance(grid, order[-1], start)
    return total

def fitness(grid, start, chromosome, items):
    order = [items[i] for i in chromosome]
    cost = route_cost(grid, start, order)
    return 1.0 / cost

def tournament_selection(population, fitnesses, k=3):
    candidates = random.sample(range(len(population)), k)
    return max(candidates, key=lambda i: fitnesses[i])

def ordered_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = p1[a:b+1]
    rem = [x for x in p2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = rem[idx]
            idx += 1
    return child

def swap_mutation(chromosome, rate=0.25):
    chrom = chromosome[:]
    if random.random() < rate:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom

def genetic_algorithm(grid, start, items, pop_size=50, generations=100):
    n = len(items)
    population = [random.sample(range(n), n) for _ in range(pop_size)]
    best_cost = float('inf')
    best_chrom = None
    history = []

    for _ in range(generations):
        fitnesses = [fitness(grid, start, c, items) for c in population]

        for chrom, fit in zip(population, fitnesses):
            cost = 1 / fit
            if cost < best_cost:
                best_cost = cost
                best_chrom = chrom[:]

        history.append(best_cost)

        sorted_pop = sorted(zip(fitnesses, population), reverse=True)
        new_pop = [x[1] for x in sorted_pop[:2]]

        while len(new_pop) < pop_size:
            p1 = population[tournament_selection(population, fitnesses)]
            p2 = population[tournament_selection(population, fitnesses)]
            child = swap_mutation(ordered_crossover(p1, p2))
            new_pop.append(child)

        population = new_pop

    return [items[i] for i in best_chrom], best_cost, history

def kmeans_zone_order(items, start, n_clusters=3):
    coords = np.array(items)
    n_clusters = min(n_clusters, len(items))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_

    start_arr = np.array(start)
    cluster_order = np.argsort(
        [np.linalg.norm(centers[k]-start_arr) for k in range(n_clusters)]
    )

    zone_order = []
    for cid in cluster_order:
        cluster_items = [items[i] for i in range(len(items)) if labels[i] == cid]
        cluster_items.sort(key=lambda p: math.dist(p, tuple(centers[cid])))
        zone_order.extend(cluster_items)

    return labels, centers, zone_order

st.title("🤖 Warehouse Robot Route Optimization")

num_items = st.sidebar.slider("Number of Items", 3, len(SHELVES), 8)
zones = st.sidebar.slider("KMeans Zones", 2, 5, 3)

if "order_items" not in st.session_state:
    st.session_state.order_items = random.sample(SHELVES, num_items)

if st.sidebar.button("Generate New Order"):
    st.session_state.order_items = random.sample(SHELVES, num_items)

ORDER_ITEMS = st.session_state.order_items

st.pyplot(
    visualize_warehouse(
        warehouse_grid,
        START,
        SHELVES,
        ORDER_ITEMS,
        title="Warehouse Layout"
    )
)

if st.button("🚀 Run Optimization"):
    ga_order, ga_cost, ga_history = genetic_algorithm(
        warehouse_grid, START, ORDER_ITEMS
    )

    naive_cost = route_cost(warehouse_grid, START, ORDER_ITEMS)

    labels, centers, km_order = kmeans_zone_order(
        ORDER_ITEMS, START, zones
    )

    km_cost = route_cost(warehouse_grid, START, km_order)

    c1, c2, c3 = st.columns(3)

    c1.metric("Naive", f"{naive_cost:.0f}")
    c2.metric("KMeans", f"{km_cost:.0f}")
    c3.metric("GA", f"{ga_cost:.0f}",
              f"{((naive_cost-ga_cost)/naive_cost)*100:.1f}%")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ga_history)
    ax.set_title("GA Convergence")
    st.pyplot(fig)

    full_path, total_cost, total_nodes = get_full_route_astar(
        warehouse_grid, START, ga_order
    )

    st.pyplot(
        visualize_warehouse(
            warehouse_grid,
            START,
            SHELVES,
            ORDER_ITEMS,
            path=[START] + full_path,
            title="GA + A* Route"
        )
    )

    df = pd.DataFrame({
        "Stop": range(1, len(ga_order)+1),
        "Shelf": [str(x) for x in ga_order]
    })

    st.dataframe(df, use_container_width=True)

    comp = pd.DataFrame({
        "Strategy":["Naive","KMeans","GA"],
        "Cost":[naive_cost, km_cost, ga_cost]
    })

    st.bar_chart(comp.set_index("Strategy"))

    st.success(
        f"Total Steps: {total_cost} | Nodes Explored: {total_nodes}"
    )
