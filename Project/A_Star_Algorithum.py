#A*算法

from PIL import Image
import numpy as np
from queue import PriorityQueue
import cv2
from env import envCube

class AStarNode:
    def __init__(self, position, g_score=float('inf'), f_score=float('inf'), parent=None):
        self.position = position
        self.g_score = g_score  # 从起始节点到当前节点的实际代价
        self.f_score = f_score  # 从起始节点到当前节点的估计代价（实际代价 + 启发式代价）
        self.parent = parent

    def __lt__(self, other):
        return self.f_score < other.f_score

def heuristic_cost_estimate(start, goal):  # 从agent到food的启发式距离
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

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

def reconstruct_path(current):
    path = []
    while current.parent:
        path.append(current.position)
        current = current.parent
    path.append(current.position)
    return path[::-1]

def is_valid_position(position, size):
    x, y = position
    return 0 <= x < size and 0 <= y < size

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def render(path, agent, food, enemies, size):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    
    for enemy in enemies:  # 绘制敌人-红色
        rect = plt.Rectangle((enemy[1], enemy[0]), 1, 1, facecolor='red')
        ax.add_patch(rect)

    rect = plt.Rectangle((agent[1], agent[0]), 1, 1, facecolor='blue')  # 绘制智能体-蓝色
    ax.add_patch(rect)

    rect = plt.Rectangle((food[1], food[0]), 1, 1, facecolor='green')  # 绘制食物-绿色
    ax.add_patch(rect)

    x = [point[1] for point in path]
    y = [point[0] for point in path]
    ax.plot(x, y, color='yellow', linewidth=1)

    plt.savefig("trajectory_3.png")  # 保存图像到文件
    plt.close(fig)


def astar_search(agent, food, enemies, size, env):
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

class A_Star:
    def train(self):
        env = envCube()
        env.reset()
        agent = (env.players[0].get_x(), env.players[0].get_y())
        food = (env.food.get_x(), env.food.get_y())
        enemies = set()
        for enemy in env.enemies:
            enemies.add((enemy.get_x(), enemy.get_y()))
        path = astar_search(agent, food, enemies, env.SIZE, env)
        print("Path:", path)
        render(path, agent, food, enemies, env.SIZE)









    