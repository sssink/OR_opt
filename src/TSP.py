import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import Model, GRB

def read_tsp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coords = []
    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            break
    for line in lines[lines.index(line) + 1:]:
        if line.startswith("EOF"):
            break
        parts = line.split()
        coords.append((float(parts[1]), float(parts[2])))

    return np.array(coords)

def calculate_distance_matrix(coords):
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return distance_matrix

def solve_tsp(distance_matrix, time_limit = 60):
    n = distance_matrix.shape[0]
    model = Model("TSP")
    model.setParam(GRB.Param.TimeLimit, time_limit)
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    model.setObjective(gb.quicksum(distance_matrix[i, j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    for i in range(n):
        model.addConstr(gb.quicksum(x[i, j] for j in range(n) if j != i) == 1)
        model.addConstr(gb.quicksum(x[j, i] for j in range(n) if j != i) == 1)
    u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")
    model.addConstrs((u[i] >= 1 for i in range(1, n)))
    model.addConstrs((u[i] <= n - 1 for i in range(1, n)))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        tour = []
        for i in range(n):
            for j in range(n):
                if x[i, j].x > 0.5:
                    tour.append((i, j))
        return model.ObjVal, tour
    else:
        return None
def plot_tsp_path(coords, tour):
    plt.figure(figsize=(10, 6))
    plt.scatter(coords[:, 0], coords[:, 1], color='blue')
    for i, j in tour:
        plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'r-')
    plt.title("TSP Path")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def convert_to_path_vector(connections):
    from collections import defaultdict
    graph = defaultdict(list)
    for a, b in connections:
        graph[a].append(b)
    path_vector = []
    visited = set()
    def dfs(node):
        visited.add(node)
        path_vector.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    for node in graph.keys():
        if node not in visited:
            dfs(node)
    return path_vector

if __name__ == "__main__":
    coords = read_tsp_instance("./data/tsp/kroA100.tsp")
    distance_matrix = calculate_distance_matrix(coords)
    Obj, tour = solve_tsp(distance_matrix, time_limit=500)
    if tour is not None:
        plot_tsp_path(coords, tour)
        print('______________OUTPUT_______________')
        print('Objective Value：', int(Obj))
        print('Solution Vector：', ' '.join(map(str,convert_to_path_vector(tour))))
    else:
        print("No solution found.")
