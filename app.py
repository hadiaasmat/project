import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import heapq
import random
import math
from sklearn.cluster import KMeans

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Warehouse Robot Optimizer", layout="wide")

random.seed(42)
np.random.seed(42)

# =====================================================
# WAREHOUSE SETUP
# =====================================================
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

# =====================================================
# VISUALIZATION
# =====================================================
def visualize_warehouse(grid, start, shelves, order_items, path=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = ListedColormap(["#F5F5F5", "#607D8B"])
    ax.imshow(grid.astype(float), cmap=cmap)

    for x in range(grid.shape[1] + 1):
        ax.axvline(x - 0.5, color="gray", linewidth=0.4)

    for y in range(grid.shape[0] + 1):
        ax.axhline(y - 0.5, color="gray", linewidth=0.4)

    for r, c in shelves:
        color = "#90CAF9"

        if (r, c) in order_items:
            color = "#FFB300"

        ax.add_patch(
            mpatches.Rectangle(
                (c - 0.35, r - 0.35),
                0.7,
                0.7,
                color=color
            )
        )

    if path:
        ax.plot(
            [p[1] for p in path],
            [p[0] for p in path],
            color="red",
            linewidth=2
        )

    ax.scatter(start[1], start[0], s=200, c="green", marker="*")

    ax.set_title("Warehouse Layout")
    return fig

# =====================================================
# A* PATHFINDING
# =====================================================
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):

    rows, cols = grid.shape

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))

    visited = {start: 0}

    while open_set:

        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path, g

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:

            nr = current[0] + dr
            nc = current[1] + dc

            if (
                0 <= nr < rows and
                0 <= nc < cols and
                grid[nr][nc] == 0
            ):

                neighbor = (nr, nc)
                new_g = g + 1

                if neighbor not in visited or new_g < visited[neighbor]:

                    visited[neighbor] = new_g

                    heapq.heappush(
                        open_set,
                        (
                            new_g + heuristic(neighbor, goal),
                            new_g,
                            neighbor,
                            path + [neighbor]
                        )
                    )

    return None, float("inf")

# =====================================================
# DISTANCE CACHE
# =====================================================
distance_cache = {}

def get_distance(grid, a, b):

    key = (a, b)

    if key not in distance_cache:

        _, cost = astar(grid, a, b)

        distance_cache[key] = cost
        distance_cache[(b, a)] = cost

    return distance_cache[key]

def route_cost(grid, start, order):

    if len(order) == 0:
        return 0

    total = get_distance(grid, start, order[0])

    for i in range(len(order)-1):
        total += get_distance(grid, order[i], order[i+1])

    total += get_distance(grid, order[-1], start)

    return total

# =====================================================
# GENETIC ALGORITHM
# =====================================================
def fitness(grid, start, chromosome, items):

    route = [items[i] for i in chromosome]
    cost = route_cost(grid, start, route)

    return 1 / cost

def ordered_crossover(p1, p2):

    size = len(p1)

    a, b = sorted(random.sample(range(size), 2))

    child = [-1] * size
    child[a:b+1] = p1[a:b+1]

    remain = [x for x in p2 if x not in child]

    idx = 0

    for i in range(size):
        if child[i] == -1:
            child[i] = remain[idx]
            idx += 1

    return child

def mutate(chromosome, rate=0.1):

    c = chromosome[:]

    if random.random() < rate:
        i, j = random.sample(range(len(c)), 2)
        c[i], c[j] = c[j], c[i]

    return c

def genetic_algorithm(items, generations=100, population_size=50):

    n = len(items)

    population = [
        random.sample(range(n), n)
        for _ in range(population_size)
    ]

    best_history = []

    for _ in range(generations):

        fitnesses = [
            fitness(
                warehouse_grid,
                START,
                p,
                items
            )
            for p in population
        ]

        best_idx = np.argmax(fitnesses)

        best_history.append(
            1 / fitnesses[best_idx]
        )

        new_population = [population[best_idx]]

        while len(new_population) < population_size:

            p1 = population[random.randint(0, population_size-1)]
            p2 = population[random.randint(0, population_size-1)]

            child = ordered_crossover(p1, p2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    fitnesses = [
        fitness(warehouse_grid, START, p, items)
        for p in population
    ]

    best_idx = np.argmax(fitnesses)

    best_order = [
        items[i]
        for i in population[best_idx]
    ]

    return best_order, best_history

# =====================================================
# K-MEANS
# =====================================================
def kmeans_order(items, start, n_clusters):

    coords = np.array(items)

    kmeans = KMeans(
        n_clusters=min(n_clusters, len(items)),
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(coords)

    centers = kmeans.cluster_centers_

    start_arr = np.array(start)

    cluster_order = np.argsort(
        [
            np.linalg.norm(c - start_arr)
            for c in centers
        ]
    )

    result = []

    for cid in cluster_order:

        cluster_items = [
            items[i]
            for i in range(len(items))
            if labels[i] == cid
        ]

        cluster_items.sort(
            key=lambda p: math.dist(
                p,
                tuple(centers[cid])
            )
        )

        result.extend(cluster_items)

    return result

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🚚 Warehouse Robot Route Optimization")

num_items = st.sidebar.slider(
    "Number of Order Items",
    3,
    12,
    8
)

clusters = st.sidebar.slider(
    "KMeans Clusters",
    2,
    5,
    3
)

if st.button("Generate Order & Optimize"):

    order_items = random.sample(SHELVES, num_items)

    naive_cost = route_cost(
        warehouse_grid,
        START,
        order_items
    )

    ga_order, history = genetic_algorithm(order_items)

    ga_cost = route_cost(
        warehouse_grid,
        START,
        ga_order
    )

    km_order = kmeans_order(
        order_items,
        START,
        clusters
    )

    km_cost = route_cost(
        warehouse_grid,
        START,
        km_order
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Naive Cost", f"{naive_cost:.0f}")
    c2.metric("K-Means Cost", f"{km_cost:.0f}")
    c3.metric("GA Cost", f"{ga_cost:.0f}")

    st.subheader("Warehouse Map")

    fig = visualize_warehouse(
        warehouse_grid,
        START,
        SHELVES,
        order_items
    )

    st.pyplot(fig)

    st.subheader("GA Convergence")

    fig2, ax = plt.subplots(figsize=(8, 4))

    ax.plot(history)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost")

    st.pyplot(fig2)

    st.subheader("Optimized Pickup Order")

    st.write(ga_order)
