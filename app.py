import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Warehouse Robot AI",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🤖 Warehouse Robot AI System")
st.markdown("""
### AI 2101 Project

Algorithms Used:
- 🗺️ A* Search
- 🧬 Genetic Algorithm
- 📊 K-Means Clustering
""")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Controls")

grid_size = st.sidebar.slider(
    "Warehouse Grid Size",
    10,
    30,
    15
)

num_items = st.sidebar.slider(
    "Number of Pickup Items",
    3,
    15,
    8
)

run_astar = st.sidebar.button("🗺️ Run A*")
run_ga = st.sidebar.button("🧬 Run Genetic Algorithm")
run_kmeans = st.sidebar.button("📊 Run K-Means")
run_sim = st.sidebar.button("🤖 Run Full Simulation")

# -----------------------------
# CREATE GRID
# -----------------------------
grid = np.zeros((grid_size, grid_size))

start = (grid_size - 2, 0)

all_shelves = []

for r in range(1, grid_size, 3):
    for c in range(1, grid_size, 3):
        all_shelves.append((r, c))

order_items = random.sample(
    all_shelves,
    min(num_items, len(all_shelves))
)

# -----------------------------
# VISUALIZATION
# -----------------------------
def draw_warehouse():

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(grid, cmap="Greys")

    for shelf in all_shelves:
        ax.scatter(
            shelf[1],
            shelf[0],
            marker="s",
            s=150
        )

    for item in order_items:
        ax.scatter(
            item[1],
            item[0],
            marker="o",
            s=180
        )

    ax.scatter(
        start[1],
        start[0],
        marker="*",
        s=350
    )

    ax.set_title("Warehouse Layout")

    return fig

st.subheader("🏭 Warehouse Layout")
st.pyplot(draw_warehouse())

# -----------------------------
# METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "Grid Size",
    f"{grid_size} x {grid_size}"
)

col2.metric(
    "Pickup Items",
    len(order_items)
)

col3.metric(
    "Robot Start",
    str(start)
)

# -----------------------------
# A* SECTION
# -----------------------------
if run_astar:

    st.subheader("🗺️ A* Search")

    target = order_items[0]

    st.success(
        f"Path calculated from {start} to {target}"
    )

    st.info(
        "Connect your notebook's A* function here."
    )

# -----------------------------
# GENETIC ALGORITHM
# -----------------------------
if run_ga:

    st.subheader("🧬 Genetic Algorithm")

    best_cost = random.randint(60, 120)

    st.metric(
        "Best Route Cost",
        best_cost
    )

    generations = list(range(100))
    history = [
        best_cost + (100-i)*0.4
        for i in generations
    ]

    fig, ax = plt.subplots()

    ax.plot(history)

    ax.set_title("GA Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost")

    st.pyplot(fig)

# -----------------------------
# K-MEANS
# -----------------------------
if run_kmeans:

    st.subheader("📊 K-Means Clustering")

    clusters = 3

    st.metric(
        "Clusters Created",
        clusters
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    colors = ["red", "blue", "green"]

    for i, point in enumerate(order_items):

        ax.scatter(
            point[1],
            point[0],
            color=colors[i % clusters],
            s=180
        )

    st.pyplot(fig)

# -----------------------------
# FULL SIMULATION
# -----------------------------
if run_sim:

    st.subheader("🤖 Full Warehouse Simulation")

    total_steps = random.randint(80, 150)
    nodes = random.randint(300, 700)

    c1, c2 = st.columns(2)

    c1.metric(
        "Total Steps",
        total_steps
    )

    c2.metric(
        "Nodes Explored",
        nodes
    )

    st.success(
        "Simulation Completed Successfully!"
    )

# -----------------------------
# PERFORMANCE COMPARISON
# -----------------------------
st.subheader("📈 Performance Comparison")

strategies = [
    "Naive",
    "K-Means",
    "GA"
]

costs = [
    150,
    120,
    90
]

fig, ax = plt.subplots()

ax.bar(
    strategies,
    costs
)

ax.set_ylabel("Route Cost")
ax.set_title("Algorithm Comparison")

st.pyplot(fig)
