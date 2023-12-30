import heapq
from env import envCube
import cv2
import numpy as np
from PIL import Image

# 定义节点类
class Node:
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

# 定义A*算法函数
def astar_search(env, start, goal):
    open_list = []
    closed_list = set()

    # 创建起始节点
    start_node = Node(start, 0, manhattan_distance(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        current_position = current_node.position

        if current_position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_list.add(current_position)

        # 获取当前节点的相邻节点
        neighbors = get_neighbors(env, current_position)

        for neighbor in neighbors:
            if neighbor in closed_list:
                continue

            g_cost = current_node.g_cost + 1
            h_cost = manhattan_distance(neighbor, goal)
            f_cost = g_cost + h_cost

            neighbor_node = Node(neighbor, g_cost, h_cost, current_node)

            if is_node_in_list(neighbor_node, open_list):
                continue

            heapq.heappush(open_list, neighbor_node)

    return None

# 获取节点的相邻节点
def get_neighbors(env, position):
    size = env.SIZE
    x, y = position
    neighbors = []

    if x > 0:
        neighbors.append((x - 1, y))
    if x < size - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < size - 1:
        neighbors.append((x, y + 1))

    return neighbors

# 计算两个位置之间的曼哈顿距离
def manhattan_distance(position1, position2):
    x1, y1 = position1
    x2, y2 = position2
    return abs(x1 - x2) + abs(y1 - y2)

# 检查节点是否在列表中
def is_node_in_list(node, node_list):
    for n in node_list:
        if n.position == node.position:
            return True
    return False

def render(self):                   #显示图片
        img = self.get_image()
        img = img.resize((800, 800))
        cv2.imshow('Predator', np.array(img))
        cv2.waitKey(1)

def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.get_x()][self.food.get_y()] = self.d[self.FOOD_N]

        for i in range(self.NUM_PLAYERS):        
            env[self.players[i].get_x()][self.players[i].get_y()] = self.d[self.PLAYER_N]

        for i in range(self.NUM_ENEMIES):
            env[self.enemies[i].get_x()][self.enemies[i].get_y()] = self.d[self.ENEMY_N]

        img = Image.fromarray(env, 'RGB')
        return img

env = envCube()
start = (0, 0)
goal = (9, 9)
path = astar_search(env, start, goal)
print("Path:", path)
