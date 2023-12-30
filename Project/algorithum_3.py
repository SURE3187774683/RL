#A*算法


import numpy as np
from queue import PriorityQueue
from env import envCube

class AStarNode:
    def __init__(self, position, g_score=float('inf'), f_score=float('inf'), parent=None):
        self.position = position
        self.g_score = g_score  #从起始节点到当前节点的实际代价
        self.f_score = f_score  #从起始节点到当前节点的估计代价（实际代价 + 启发式代价）
        self.parent = parent

    def __lt__(self, other):
        return self.f_score < other.f_score

def heuristic_cost_estimate(start, goal):#从agent到food的启发式距离
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def reconstruct_path(current):
    path = []
    while current.parent:
        path.append(current.position)
        current = current.parent
    path.append(current.position)
    return path[::-1]

def action_to_delta(action):
    if action == 0:
        return (0, 1)
    elif action == 1:
        return (0, -1)
    elif action == 2:
        return (1, 0)
    elif action == 3:
        return (-1, 0)
    elif action == 4:
        return (1, 1)
    elif action == 5:
        return (-1, 1)
    elif action == 6:
        return (1, -1)
    elif action == 7:
        return (-1, -1)
    elif action == 8:
        return (0, 0)

def is_valid_position(position, size):
    x, y = position
    return 0 <= x < size and 0 <= y < size

def astar_search(agent, food, enemies, size):
    open_set = PriorityQueue()
    start_node = AStarNode(agent, 0, heuristic_cost_estimate(agent, food))
    open_set.put(start_node)

    closed_set = set()
    g_scores = {agent: 0}

    while not open_set.empty():
        current_node = open_set.get()
        current_position = current_node.position

        if current_position == food:
            return reconstruct_path(current_node)

        closed_set.add(current_position)

        for action in range(9):
            next_position = tuple(np.add(current_position, action_to_delta(action)))

            if not is_valid_position(next_position, size) or next_position in enemies:
                continue

            g_score = g_scores[current_position] + 1
            if g_score < g_scores.get(next_position, float('inf')):
                g_scores[next_position] = g_score
                f_score = g_score + heuristic_cost_estimate(next_position, food)
                neighbor = AStarNode(next_position, g_score, f_score, current_node)
                open_set.put(neighbor)

    return None

env = envCube()
state = env.reset()
agent = (env.players[0].get_x(), env.players[0].get_y())
food = (env.food.get_x(), env.food.get_y())
enemies = set()
for enemy in env.enemies:
    enemies.add((enemy.get_x(), enemy.get_y()))
path = astar_search(agent, food, enemies, env.SIZE) # 使用A*算法进行路径规划

if path is not None:    # 输出路径
    print("Path:", path)
else:
    print("No path found.")

