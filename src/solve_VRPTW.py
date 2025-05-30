from gurobipy import Model, GRB, quicksum
from sko.GA import GA
from pathlib import Path
import numpy as np
import time
import pandas as pd

class VRPTW_Instance:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vehicle_num, self.capacity, self.customer_num, self.x_coord, self.y_coord, self.demand, self.ready_time, self.due_time, self.service_time = self.read_vrptw_instance()
        self.distance_matrix = self.calculate_distance_matrix()
        self.node_num = self.customer_num + 2
        self.vehicle_velocity = 1

    def read_vrptw_instance(self):
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        # Skip header lines until we find VEHICLE
        for i, line in enumerate(lines):
            if line.strip().startswith('VEHICLE'):
                break
        
        # Read vehicle info
        vehicle_info = lines[i+2].split()
        vehicle_num = int(vehicle_info[0])
        capacity = int(vehicle_info[1])

        # Skip to customer data
        for i, line in enumerate(lines):
            if line.strip().startswith('CUSTOMER'):
                break
        
        # Initialize lists to store customer data
        x_coord = []
        y_coord = []
        demand = []
        ready_time = []
        due_time = []
        service_time = []

        # Read customer data
        for line in lines[i+3:]:
            if not line.strip():
                break
            data = line.split()
            x_coord.append(float(data[1]))
            y_coord.append(float(data[2]))
            demand.append(int(data[3]))
            ready_time.append(int(data[4]))
            due_time.append(int(data[5]))
            service_time.append(int(data[6]))

        x_coord.append(x_coord[0])
        y_coord.append(y_coord[0])
        demand.append(demand[0])
        ready_time.append(ready_time[0])
        due_time.append(due_time[0])
        service_time.append(service_time[0])

        # Convert lists to numpy arrays
        x_coord = np.array(x_coord)
        y_coord = np.array(y_coord)
        demand = np.array(demand)
        ready_time = np.array(ready_time)
        due_time = np.array(due_time)
        service_time = np.array(service_time)

        customer_num = len(x_coord)-2
        return vehicle_num, capacity, customer_num, x_coord, y_coord, demand, ready_time, due_time, service_time

    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.customer_num, self.customer_num))
        for i in range(self.customer_num):
            for j in range(i+1, self.customer_num):
                distance_matrix[i, j] = np.sqrt((self.x_coord[i] - self.x_coord[j])**2 + (self.y_coord[i] - self.y_coord[j])**2)
                distance_matrix[j, i] = distance_matrix[i, j]
        return distance_matrix

def solve_vrptw_gurobi(instance, time_limit = 1200):
    # 创建模型
    model = Model('VRPTW')
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam('OutputFlag', 0)
    # 添加决策变量
    # x[i,j,k] 表示车辆k是否从客户i到客户j
    x = model.addVars(instance.node_num, instance.node_num, instance.vehicle_num, vtype=GRB.BINARY, name='x')
    # t[i,k] 表示车辆k到达客户i的时间
    t = model.addVars(instance.node_num, instance.vehicle_num, vtype=GRB.CONTINUOUS, name='t')
    # y[k] 表示是否使用车辆k
    y = model.addVars(instance.vehicle_num, vtype=GRB.BINARY, name='y')

    # 目标函数：优先最小化车辆数量，其次最小化总行驶距离
    # 使用大系数M来确保车辆数量最小化是第一优先级
    M = 10000
    model.setObjective(M * quicksum(y[k] for k in range(instance.vehicle_num)) +
                      quicksum(instance.distance_matrix[i,j] * x[i,j,k] 
                             for i in range(instance.node_num) 
                             for j in range(instance.node_num) 
                             for k in range(instance.vehicle_num)), GRB.MINIMIZE)

    # 约束条件
    # 1. 每个客户必须且只能被一辆车访问一次
    for i in range(1, instance.node_num-1):
        model.addConstr(quicksum(x[i,j,k] 
                               for j in range(instance.node_num) 
                               for k in range(instance.vehicle_num) if i != j) == 1)

    # 2. 每辆车的容量约束
    for k in range(instance.vehicle_num):
        model.addConstr(quicksum(instance.demand[i] * x[i,j,k] 
                               for i in range(1, instance.node_num-1)
                               for j in range(instance.node_num) if i != j) <= instance.capacity * y[k])

    # 3. 车辆流平衡约束
    for h in range(1, instance.node_num-1):
        for k in range(instance.vehicle_num):
            model.addConstr(quicksum(x[i,h,k] for i in range(instance.node_num) if i != h) == 
                          quicksum(x[h,j,k] for j in range(instance.node_num) if j != h))

    # 4. 所有使用的车辆从仓库出发并返回仓库
    for k in range(instance.vehicle_num):
        model.addConstr(quicksum(x[0,j,k] for j in range(1, instance.node_num-1)) == y[k])
        model.addConstr(quicksum(x[i,instance.node_num-1,k] for i in range(1, instance.node_num-1)) == y[k])

    # 5. 时间窗约束
    BIG_M = 10000  # 一个足够大的数
    for i in range(instance.node_num):
        for j in range(instance.node_num):
            if i != j:
                for k in range(instance.vehicle_num):
                    # 考虑车辆速度的行驶时间计算
                    travel_time = instance.distance_matrix[i,j] / instance.vehicle_velocity
                    # 如果车辆k从i到j，则到达j的时间应该满足时间窗约束
                    model.addConstr(t[i,k] + instance.service_time[i] + 
                                  travel_time - BIG_M * (1 - x[i,j,k]) <= t[j,k])
                    model.addConstr(t[j,k] >= instance.ready_time[j])
                    model.addConstr(t[j,k] <= instance.due_time[j])

    # 6. 确保未使用的车辆不参与配送
    for k in range(instance.vehicle_num):
        for i in range(instance.node_num):
            for j in range(instance.node_num):
                if i != j:
                    model.addConstr(x[i,j,k] <= y[k])

    # 求解模型
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = []
        used_vehicles = sum(y[k].x > 0.5 for k in range(instance.vehicle_num))
        total_distance = sum(instance.distance_matrix[i,j] * x[i,j,k].x 
                           for i in range(instance.node_num)
                           for j in range(instance.node_num)
                           for k in range(instance.vehicle_num))
        for k in range(instance.vehicle_num):
            if y[k].x > 0.5:  # 只输出使用的车辆的路径
                for i in range(instance.node_num):
                    for j in range(instance.node_num):
                        if i != j and x[i,j,k].x > 0.5:
                            solution.append((i,j,k))
        return model.objVal, solution, model.Runtime, used_vehicles, total_distance
    else:
        return None, None, None, None, None

def solve_vrptw_GA(m, n, capacities, weights, profits):
    # 定义目标函数
    def objective_function(x):
        x_binary = np.where(x > 0.5, 1, 0)
        total_profit = 0
        for i in range(m):
            for j in range(n):
                total_profit += profits[j] * x_binary[i * n + j]  # 正确计算索引
        return total_profit  # 由于scikit-opt是最小化，我们取负值来最大化利润
    # 定义约束条件的惩罚函数
    def penalty(x):
        x_binary = np.where(x > 0.5, 1, 0)
        penalty_value = 0
        for i in range(m):
            weight_sum = sum(weights[j] * x_binary[i * n + j] for j in range(n))
            if weight_sum > capacities[i]:
                penalty_value += (weight_sum - capacities[i]) ** 2 * 1000
        for j in range(n):
            quantity_sum = sum(x_binary[i * n + j] for i in range(m))
            if quantity_sum > 1:
                penalty_value += (quantity_sum - 1) ** 2 * 100000
        return penalty_value
    # 定义变量的界限
    lb = [0] * (m * n)
    ub = [1] * (m * n)
    # 初始化GA算法
    ga = GA(func=lambda x: -objective_function(x) + penalty(x), n_dim=m * n, size_pop=200, max_iter=1200, lb=lb, ub=ub, precision=1)
    # 记录开始时间
    start_time = time.time()
    # 运行GA算法
    ga.run()
    # 记录结束时间并计算求解时间
    end_time = time.time()
    solve_time = end_time - start_time
    max_profit = objective_function(ga.best_x)
    # max_profit = objective_function(ga.best_x)
    # 将实数解转换为二进制解
    best_x_binary = np.where(ga.best_x > 0.5, 1, 0)
    return max_profit, best_x_binary.reshape(m,n), solve_time


if __name__ == "__main__":
    # data_folder_path = Path("./data/VRPTW/200")
    # df_gurobi = pd.DataFrame(columns=["dataset", "best_value", "solution", "solve_time"])
    # # df_ga = pd.DataFrame(columns=["dataset", "best_value", "solution", "solve_time"])
    # i = 0
    # for data_file_path in data_folder_path.glob("*.txt"):
    #     m, n, capacities, weights, profits = read_vrptw_instance(data_file_path)
    #     Obj, solution, t = solve_vrptw_gurobi(m, n, capacities, weights, profits, time_limit=600)
    #     print('data_file:', data_file_path)
    #     print('______________GUROBI_OUTPUT_______________')
    #     if solution is not None:
    #         print('Objective Value：', Obj)
    #         print('Solution Vector：', solution)
    #         print('Solve Time：', t)
    #         df_gurobi.loc[i] = [data_file_path,Obj,solution,t]
    #     else:
    #         print("No solution found.")
    #         df_gurobi.loc[i] = [data_file_path, None, None, None]
    #     print('_______________GUROBI_END_________________')
    #     # Obj, solution, t = solve_vrptw_GA(m, n, capacities, weights, profits)
    #     # print('______________GA_OUTPUT_______________')
    #     # print('Objective Value：', Obj)
    #     # print('Solution Vector：', solution)
    #     # print('Solve Time：', t)
    #     # df_ga.loc[i] = [data_file_path, Obj, solution, t]
    #     # print('_______________GA_END_________________')
    #     i += 1
    #     if i == 5:
    #         break
    # df_gurobi.to_csv('sol_gurobi.csv')
    # # df_ga.to_csv('sol_ga.csv')
    instance = VRPTW_Instance("./data/VRPTW/200/C1_2_1.TXT")
    print(instance.vehicle_num, instance.capacity, instance.customer_num, instance.x_coord, instance.y_coord, instance.demand, instance.ready_time, instance.due_time, instance.service_time)
    print(instance.distance_matrix)

