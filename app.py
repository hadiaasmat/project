import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import random
import heapq
import math
from sklearn.cluster import KMeans

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Warehouse Robot Optimizer",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background-color:#f8fafc;
}

.metric-box{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

h1,h2,h3{
    color:#0f172a;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# WAREHOUSE
# ==========================================================
GRID_SIZE = 15

warehouse_grid = np.zeros((GRID_SIZE, GRID_SIZE))

rack_cols = [2,5,8,11]
rack_rows = [(1,5),(7,11),(1,5),(7,11)]

for col,(rs,re) in zip(rack_cols,rack_rows):
    for r in range(rs,re+1):
        warehouse_grid[r][col] = 1

START = (13,0)

SHELVES = [
    (1,1),(1,4),(1,7),(1,10),(1,13),
    (4,1),(4,4),(4,7),(4,10),(4,13),
    (8,1),(8,4),(8,7),(8,10),(8,13),
    (11,1),(11,4),(11,7),(11,10),(11,13)
]

# ==========================================================
# A*
# ==========================================================
def heuristic(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid,start,goal):

    rows, cols = grid.shape

    open_set = []
    heapq.heappush(
        open_set,
        (heuristic(start,goal),0,start,[start])
    )

    visited = {start:0}

    while open_set:

        f,g,current,path = heapq.heappop(open_set)

        if current == goal:
            return path,g

        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:

            nr = current[0]+dr
            nc = current[1]+dc

            if (
                0 <= nr < rows and
                0 <= nc < cols and
                grid[nr][nc] == 0
            ):

                neighbor = (nr,nc)
                new_g = g + 1

                if neighbor not in visited or new_g < visited[neighbor]:

                    visited[neighbor] = new_g

                    heapq.heappush(
                        open_set,
                        (
                            new_g + heuristic(neighbor,goal),
                            new_g,
                            neighbor,
                            path+[neighbor]
                        )
                    )

    return None,float("inf")

distance_cache = {}

def get_distance(a,b):

    key = (a,b)

    if key not in distance_cache:

        _,cost = astar(
            warehouse_grid,
            a,
            b
        )

        distance_cache[key] = cost
        distance_cache[(b,a)] = cost

    return distance_cache[key]

def route_cost(route):

    if len(route)==0:
        return 0

    total = get_distance(START,route[0])

    for i in range(len(route)-1):
        total += get_distance(
            route[i],
            route[i+1]
        )

    total += get_distance(
        route[-1],
        START
    )

    return total

# ==========================================================
# GA
# ==========================================================
def fitness(chromosome,items):

    route = [items[i] for i in chromosome]

    cost = route_cost(route)

    return 1/cost

def crossover(p1,p2):

    size = len(p1)

    a,b = sorted(
        random.sample(
            range(size),
            2
        )
    )

    child = [-1]*size

    child[a:b+1] = p1[a:b+1]

    remain = [x for x in p2 if x not in child]

    idx = 0

    for i in range(size):
        if child[i] == -1:
            child[i] = remain[idx]
            idx += 1

    return child

def mutate(chromosome,rate=0.1):

    chromosome = chromosome[:]

    if random.random() < rate:

        i,j = random.sample(
            range(len(chromosome)),
            2
        )

        chromosome[i],chromosome[j] = (
            chromosome[j],
            chromosome[i]
        )

    return chromosome

def genetic_algorithm(
    items,
    generations=100,
    population_size=50
):

    n = len(items)

    population = [
        random.sample(range(n),n)
        for _ in range(population_size)
    ]

    history = []

    for _ in range(generations):

        scores = [
            fitness(p,items)
            for p in population
        ]

        best_idx = np.argmax(scores)

        history.append(
            1/scores[best_idx]
        )

        new_population = [
            population[best_idx]
        ]

        while len(new_population) < population_size:

            p1 = random.choice(population)
            p2 = random.choice(population)

            child = crossover(p1,p2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    scores = [
        fitness(p,items)
        for p in population
    ]

    best_idx = np.argmax(scores)

    best_route = [
        items[i]
        for i in population[best_idx]
    ]

    return best_route,history

# ==========================================================
# KMEANS
# ==========================================================
def kmeans_route(items,k=3):

    coords = np.array(items)

    km = KMeans(
        n_clusters=min(k,len(items)),
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(coords)

    centers = km.cluster_centers_

    cluster_order = np.argsort([
        np.linalg.norm(
            center - np.array(START)
        )
        for center in centers
    ])

    result = []

    for cid in cluster_order:

        group = [
            items[i]
            for i in range(len(items))
            if labels[i]==cid
        ]

        result.extend(group)

    return result

# ==========================================================
# VISUALIZATION
# ==========================================================
def warehouse_plot(order_items,path_items=None):

    fig, ax = plt.subplots(figsize=(8,8))

    cmap = ListedColormap(
        ["white","#607d8b"]
    )

    ax.imshow(
        warehouse_grid,
        cmap=cmap
    )

    for shelf in SHELVES:

        color = "#90caf9"

        if shelf in order_items:
            color = "#ff9800"

        ax.add_patch(
            patches.Rectangle(
                (shelf[1]-0.35,shelf[0]-0.35),
                0.7,
                0.7,
                color=color
            )
        )

    ax.scatter(
        START[1],
        START[0],
        c="green",
        s=200,
        marker="*"
    )

    if path_items:

        xs = [p[1] for p in path_items]
        ys = [p[0] for p in path_items]

        ax.plot(xs,ys,linewidth=3)

    ax.set_title(
        "Warehouse Layout"
    )

    return fig

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:

    st.title("🤖 Warehouse AI")

    num_items = st.slider(
        "Order Items",
        3,
        12,
        8
    )

    generations = st.slider(
        "GA Generations",
        50,
        300,
        150
    )

    population_size = st.slider(
        "Population Size",
        20,
        150,
        60
    )

    clusters = st.slider(
        "KMeans Clusters",
        2,
        5,
        3
    )

    run = st.button(
        "🚀 Optimize Route",
        use_container_width=True
    )

# ==========================================================
# TITLE
# ==========================================================
st.title("🚚 Warehouse Robot Route Optimization Dashboard")

# ==========================================================
# RUN
# ==========================================================
if run:

    order_items = random.sample(
        SHELVES,
        num_items
    )

    naive_cost = route_cost(order_items)

    km_route = kmeans_route(
        order_items,
        clusters
    )

    km_cost = route_cost(km_route)

    ga_route,history = genetic_algorithm(
        order_items,
        generations,
        population_size
    )

    ga_cost = route_cost(
        ga_route
    )

    improvement = (
        (naive_cost-ga_cost)
        / naive_cost
    )*100

    # ======================================================
    # TABS
    # ======================================================
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "🏠 Dashboard",
        "🗺 Warehouse",
        "📊 Comparison",
        "🧬 GA Evolution",
        "📋 Orders"
    ])

    # ======================================================
    # DASHBOARD
    # ======================================================
    with tab1:

        c1,c2,c3,c4 = st.columns(4)

        c1.metric(
            "Naive Cost",
            f"{naive_cost:.0f}"
        )

        c2.metric(
            "KMeans Cost",
            f"{km_cost:.0f}"
        )

        c3.metric(
            "GA Cost",
            f"{ga_cost:.0f}"
        )

        c4.metric(
            "Improvement",
            f"{improvement:.2f}%"
        )

        st.success(
            f"GA reduced travel distance by {improvement:.2f}%"
        )

    # ======================================================
    # WAREHOUSE
    # ======================================================
    with tab2:

        st.pyplot(
            warehouse_plot(
                order_items,
                ga_route
            )
        )

    # ======================================================
    # COMPARISON
    # ======================================================
    with tab3:

        df = pd.DataFrame({
            "Method":[
                "Naive",
                "KMeans",
                "Genetic Algorithm"
            ],
            "Cost":[
                naive_cost,
                km_cost,
                ga_cost
            ]
        })

        st.dataframe(
            df,
            use_container_width=True
        )

        st.bar_chart(
            df.set_index("Method")
        )

    # ======================================================
    # GA EVOLUTION
    # ======================================================
    with tab4:

        fig,ax = plt.subplots(
            figsize=(10,4)
        )

        ax.plot(history)

        ax.set_xlabel(
            "Generation"
        )

        ax.set_ylabel(
            "Best Route Cost"
        )

        ax.grid(True)

        st.pyplot(fig)

    # ======================================================
    # ORDERS
    # ======================================================
    with tab5:

        st.subheader(
            "Order Items"
        )

        order_df = pd.DataFrame(
            order_items,
            columns=[
                "Row",
                "Column"
            ]
        )

        st.dataframe(
            order_df,
            use_container_width=True
        )

        st.subheader(
            "Optimized Sequence"
        )

        route_df = pd.DataFrame(
            ga_route,
            columns=[
                "Row",
                "Column"
            ]
        )

        st.dataframe(
            route_df,
            use_container_width=True
        )

        csv = route_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Route CSV",
            csv,
            "optimized_route.csv",
            "text/csv"
        )

else:

    st.info(
        "Select parameters from the sidebar and click 'Optimize Route'."
    )
