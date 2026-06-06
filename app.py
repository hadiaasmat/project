import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import random
import heapq
import math

from sklearn.cluster import KMeans

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Warehouse Robot AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Warehouse Robot AI System")
st.markdown("### A* Search • Genetic Algorithm • K-Means Clustering")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("⚙️ Configuration")

num_items = st.sidebar.slider(
    "Order Items",
    3,
    12,
    8
)

ga_generations = st.sidebar.slider(
    "GA Generations",
    20,
    300,
    100
)

ga_population = st.sidebar.slider(
    "Population Size",
    10,
    100,
    50
)

num_clusters = st.sidebar.slider(
    "K-Means Zones",
    2,
    5,
    3
)

# ==================================================
# WAREHOUSE SETUP
# ==================================================

GRID_SIZE = 15

warehouse_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

rack_cols = [2, 5, 8, 11]
rack_rows = [(1, 5), (7, 11), (1, 5), (7, 11)]

for col, (r_start, r_end) in zip(rack_cols, rack_rows):
    for r in range(r_start, r_end + 1):
        warehouse_grid[r][col] = 1

START = (13, 0)

SHELVES = [
    (1,1),(1,4),(1,7),(1,10),(1,13),
    (4,1),(4,4),(4,7),(4,10),(4,13),
    (8,1),(8,4),(8,7),(8,10),(8,13),
    (11,1),(11,4),(11,7),(11,10),(11,13),
]

ORDER_ITEMS = random.sample(SHELVES, num_items)

# ==================================================
# VISUALIZATION
# ==================================================

def visualize_warehouse(grid,start,shelves,order_items,path=None,
                        title="Warehouse",cluster_labels=None):

    fig, ax = plt.subplots(figsize=(8,8))

    cmap = ListedColormap(['#F5F5F5','#607D8B'])

    ax.imshow(grid.astype(float), cmap=cmap)

    for x in range(grid.shape[1]+1):
        ax.axvline(x-0.5,color='lightgray',linewidth=0.5)

    for y in range(grid.shape[0]+1):
        ax.axhline(y-0.5,color='lightgray',linewidth=0.5)

    cluster_colors = [
        '#E53935',
        '#43A047',
        '#1E88E5',
        '#FB8C00',
        '#8E24AA'
    ]

    for (r,c) in shelves:

        color = "#90CAF9"

        if cluster_labels is not None and (r,c) in order_items:
            idx = order_items.index((r,c))
            color = cluster_colors[
                cluster_labels[idx] %
                len(cluster_colors)
            ]

        elif (r,c) in order_items:
            color = "#FFB300"

        ax.add_patch(
            mpatches.Rectangle(
                (c-0.35,r-0.35),
                0.7,
                0.7,
                color=color
            )
        )

    if path:
        ax.plot(
            [p[1] for p in path],
            [p[0] for p in path],
            linewidth=3
        )

    ax.scatter(
        start[1],
        start[0],
        s=250,
        marker="*"
    )

    ax.set_title(title)

    return fig

# ==================================================
# A STAR
# ==================================================

def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def astar(grid,start,goal):

    rows, cols = grid.shape

    open_set=[]

    heapq.heappush(
        open_set,
        (
            heuristic(start,goal),
            0,
            start,
            [start]
        )
    )

    g_scores={start:0}

    nodes=0

    while open_set:

        f,g,current,path = heapq.heappop(open_set)

        nodes += 1

        if current == goal:
            return path,g,nodes

        for dr,dc in [
            (-1,0),
            (1,0),
            (0,-1),
            (0,1)
        ]:

            nr=current[0]+dr
            nc=current[1]+dc

            if (
                0<=nr<rows and
                0<=nc<cols and
                grid[nr][nc]==0
            ):

                neighbor=(nr,nc)

                new_g=g+1

                if (
                    neighbor not in g_scores
                    or
                    new_g<g_scores[neighbor]
                ):

                    g_scores[neighbor]=new_g

                    heapq.heappush(
                        open_set,
                        (
                            new_g+
                            heuristic(neighbor,goal),
                            new_g,
                            neighbor,
                            path+[neighbor]
                        )
                    )

    return None,float("inf"),nodes

# ==================================================
# DISTANCE CACHE
# ==================================================

distance_cache = {}

def get_distance(grid,a,b):

    key=(a,b)

    if key not in distance_cache:

        _,cost,_=astar(grid,a,b)

        distance_cache[key]=cost
        distance_cache[(b,a)]=cost

    return distance_cache[key]

def route_cost(grid,start,order):

    total=get_distance(grid,start,order[0])

    for i in range(len(order)-1):
        total += get_distance(
            grid,
            order[i],
            order[i+1]
        )

    total += get_distance(
        grid,
        order[-1],
        start
    )

    return total

# ==================================================
# GENETIC ALGORITHM
# ==================================================

def fitness(grid,start,chromosome,items):

    order=[items[i] for i in chromosome]

    cost=route_cost(
        grid,
        start,
        order
    )

    return 1/cost

def tournament_selection(pop,fitnesses,k=3):

    candidates=random.sample(
        range(len(pop)),
        k
    )

    return max(
        candidates,
        key=lambda i: fitnesses[i]
    )

def ordered_crossover(p1,p2):

    size=len(p1)

    a,b=sorted(
        random.sample(
            range(size),
            2
        )
    )

    child=[-1]*size

    child[a:b+1]=p1[a:b+1]

    remaining=[
        x for x in p2
        if x not in child
    ]

    idx=0

    for i in range(size):

        if child[i]==-1:

            child[i]=remaining[idx]
            idx+=1

    return child

def swap_mutation(chrom,rate=0.25):

    chrom=chrom[:]

    if random.random()<rate:

        i,j=random.sample(
            range(len(chrom)),
            2
        )

        chrom[i],chrom[j]=chrom[j],chrom[i]

    return chrom

def genetic_algorithm(
        grid,
        start,
        items,
        pop_size,
        generations
):

    n=len(items)

    population=[
        random.sample(range(n),n)
        for _ in range(pop_size)
    ]

    best_history=[]

    best_cost=float("inf")
    best_chrom=None

    for _ in range(generations):

        fitnesses=[
            fitness(grid,start,c,items)
            for c in population
        ]

        for chrom,fit in zip(
            population,
            fitnesses
        ):

            cost=1/fit

            if cost<best_cost:

                best_cost=cost
                best_chrom=chrom[:]

        best_history.append(best_cost)

        new_pop=[]

        while len(new_pop)<pop_size:

            p1=population[
                tournament_selection(
                    population,
                    fitnesses
                )
            ]

            p2=population[
                tournament_selection(
                    population,
                    fitnesses
                )
            ]

            child=ordered_crossover(
                p1,p2
            )

            child=swap_mutation(child)

            new_pop.append(child)

        population=new_pop

    best_order=[
        items[i]
        for i in best_chrom
    ]

    return best_order,best_cost,best_history

# ==================================================
# KMEANS
# ==================================================

def kmeans_zone_order(
    items,
    start,
    n_clusters
):

    coords=np.array(items)

    kmeans=KMeans(
        n_clusters=min(
            n_clusters,
            len(items)
        ),
        random_state=42,
        n_init=10
    )

    labels=kmeans.fit_predict(coords)

    centers=kmeans.cluster_centers_

    return labels,centers

# ==================================================
# KPI
# ==================================================

c1,c2,c3,c4 = st.columns(4)

c1.metric("Grid",f"{GRID_SIZE}x{GRID_SIZE}")
c2.metric("Shelves",len(SHELVES))
c3.metric("Items",len(ORDER_ITEMS))
c4.metric("Start",str(START))

# ==================================================
# TABS
# ==================================================

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🏭 Warehouse",
    "🗺️ A*",
    "🧬 GA",
    "📊 K-Means",
    "🤖 Simulation"
])

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.pyplot(
        visualize_warehouse(
            warehouse_grid,
            START,
            SHELVES,
            ORDER_ITEMS,
            title="Warehouse Layout"
        )
    )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    target=ORDER_ITEMS[0]

    path,cost,nodes=astar(
        warehouse_grid,
        START,
        target
    )

    col1,col2=st.columns(2)

    col1.metric(
        "Path Length",
        cost
    )

    col2.metric(
        "Nodes Explored",
        nodes
    )

    st.pyplot(
        visualize_warehouse(
            warehouse_grid,
            START,
            SHELVES,
            ORDER_ITEMS,
            path,
            "A* Pathfinding"
        )
    )

# ==================================================
# TAB 3
# ==================================================

with tab3:

    ga_order,ga_cost,ga_history=genetic_algorithm(
        warehouse_grid,
        START,
        ORDER_ITEMS,
        ga_population,
        ga_generations
    )

    naive_cost=route_cost(
        warehouse_grid,
        START,
        ORDER_ITEMS
    )

    improvement=(
        (naive_cost-ga_cost)
        /
        naive_cost
    )*100

    st.metric(
        "Improvement %",
        f"{improvement:.1f}%"
    )

    fig,ax=plt.subplots()

    ax.plot(ga_history)

    ax.set_title(
        "GA Convergence"
    )

    st.pyplot(fig)

# ==================================================
# TAB 4
# ==================================================

with tab4:

    labels,centers=kmeans_zone_order(
        ORDER_ITEMS,
        START,
        num_clusters
    )

    st.pyplot(
        visualize_warehouse(
            warehouse_grid,
            START,
            SHELVES,
            ORDER_ITEMS,
            cluster_labels=labels
        )
    )

# ==================================================
# TAB 5
# ==================================================

with tab5:

    labels,centers=kmeans_zone_order(
        ORDER_ITEMS,
        START,
        num_clusters
    )

    km_cost=route_cost(
        warehouse_grid,
        START,
        ORDER_ITEMS
    )

    results=pd.DataFrame({
        "Strategy":[
            "Naive",
            "KMeans",
            "GA"
        ],
        "Cost":[
            naive_cost,
            km_cost,
            ga_cost
        ]
    })

    st.dataframe(
        results,
        use_container_width=True
    )

    fig,ax=plt.subplots()

    ax.bar(
        results["Strategy"],
        results["Cost"]
    )

    ax.set_ylabel(
        "Total Steps"
    )

    ax.set_title(
        "Performance Comparison"
    )

    st.pyplot(fig)

    st.download_button(
        "📥 Download Results",
        results.to_csv(index=False),
        "warehouse_results.csv"
    )
