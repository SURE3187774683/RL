# grid的地图中agent躲避enemy，想吃到food
# agent、enemy的数量可自定义
# 静态和动态环境可选择
# 奖惩机制可自定义
# SURE

import numpy as np
import matplotlib.pyplot as plt

ENV_MOVE = False                            #env是否变化
MAX_STEP = 350                              #每局最大步数

# 建立Cube类，用于创建player、food和enemy
class Cube:
    def __init__(self, size, x=None, y=None):  # 生成个体的位置
        self.size = size
        if x is not None and y is not None:
            self.x = x
            self.y = y
        else:
            self.x = np.random.randint(0, self.size)
            self.y = np.random.randint(0, self.size)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __str__(self):  # 将位置转化为字符串形式
        return f'{self.x},{self.y}'

    def __sub__(self, other):  # 位置相减（subtraction）
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):  # 判断两个智能体位置是否相同
        return self.x == other.x and self.y == other.y

    def action(self, choise):  # action函数（向8个位置移动或静止）
        if choise == 0:
            self.move(x=0, y=1)
        elif choise == 1:
            self.move(x=0, y=-1)
        elif choise == 2:
            self.move(x=1, y=0)
        elif choise == 3:
            self.move(x=-1, y=0)
        elif choise == 4:
            self.move(x=1, y=1)
        elif choise == 5:
            self.move(x=-1, y=1)
        elif choise == 6:
            self.move(x=1, y=-1)
        elif choise == 7:
            self.move(x=-1, y=-1)
        elif choise == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):  # 移动函数
        if not x:  # x，y未定义时随意指定
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
            
        if self.x < 0:  # 检测环境边界
            self.x = 0
        elif self.x >= self.size:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y >= self.size:
            self.y = self.size - 1

class envCube:  # 生成环境类
    SIZE = 30         #地图大小
    ACTION_SPACE_VALUES = 9 #action的数量
    FOOD_REWARD = 100
    ENEMY_PENALITY = -10
    MOVE_PENALITY = -4
    CLOSER_REWARD = 5
    FARER_PENALITY = -5
    STAY_REWARD = 0

    def __init__(self,player_positions,enemy_positions) -> None:
        self.enemy_positions = enemy_positions
        self.player_positions = player_positions
        self.NUM_PLAYERS = len(player_positions)    # player的数量
        self.NUM_ENEMIES = len(enemy_positions)     # enemy的数量
        self.OBSERVATION_SPACE_VALUES = (2+2*self.NUM_ENEMIES)*self.NUM_PLAYERS  # state的数量

    def reset(self):
        self.trajectory = [[] for _ in range(self.NUM_PLAYERS)]          # 在每个步骤开始之前清空轨迹列表
        
        self.old_distances = float('inf')  # 初始化为正无穷大

        self.food = Cube(self.SIZE,23,23)       # 创建food

        self.players = []                       # 创建players
        for i in range(self.NUM_PLAYERS):
            x, y = self.player_positions[i]  
            self.player = Cube(self.SIZE, x, y)  
            self.players.append(self.player)

        self.enemies = []                       # 创建enemy
        for i in range(self.NUM_ENEMIES):
            x, y = self.enemy_positions[i]  
            self.enemy = Cube(self.SIZE, x, y)  
            self.enemies.append(self.enemy)

        state = ()                              # 记录状态
        for i in range(self.NUM_PLAYERS):
            state += (self.players[i] - self.food)
            for j in range(self.NUM_ENEMIES):
                state += (self.players[i] - self.enemies[j])
        self.episode_step = 0

        return state

    def step(self, player_id,action):
        equal_p_e = False
        self.episode_step += 1        

        self.players[player_id].action(action)

        if ENV_MOVE == True:
            self.food.move()
            for enemy in self.enemies:
                enemy.move()
   
        new_observation = () 

        new_distances = [] #下一步的每个agent和food的距离之和
        for i in range(self.NUM_PLAYERS):
            new_observation += (self.players[i] - self.food)    # 更新state
            distance = np.linalg.norm(new_observation, ord=1)   # 计算代理和食物的距离
            new_distances.append(distance)
            
            for j in range(self.NUM_ENEMIES):
                new_observation += (self.players[i] - self.enemies[j])

        new_distances_sum = np.sum(new_distances)

        if self.old_distances>new_distances_sum:
            reward = self.CLOSER_REWARD
        elif self.old_distances<new_distances_sum:
            reward = self.FARER_PENALITY
        else:
            reward = self.STAY_REWARD
        
        self.old_distances = new_distances

        for i in range(self.NUM_PLAYERS):           #定义奖励机制
            for j in range(self.NUM_ENEMIES):
                if self.players[i] == self.enemies[j]:
                    equal_p_e = True

            if self.players[i] == self.food:
                reward += self.FOOD_REWARD
            elif equal_p_e:
                reward += self.ENEMY_PENALITY
            else:
                reward += self.MOVE_PENALITY
        done = False

        #将智能体的位置添加到轨迹列表中
        for i in range(self.NUM_PLAYERS):
            self.trajectory[i].append((self.players[i].get_x(), self.players[i].get_y()))

        #任意一个玩家被吃掉/都到达/超过200步，游戏结束
        for i in range(self.NUM_PLAYERS):
            for j in range(self.NUM_ENEMIES):
                if self.players[i] == self.food or self.players[i] == self.enemies[j] or self.episode_step >= MAX_STEP:
                    done = True

        #任意两个玩家相撞，游戏结束
        for i in range(self.NUM_PLAYERS):
            for j in range(i+1, self.NUM_PLAYERS):
                if self.players[i].get_x() == self.players[j].get_x() and self.players[i].get_y() == self.players[j].get_y():
                    done = True

        return new_observation, reward, done
        
    def render_trajectory(self, flag):          #记录agent轨迹
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.SIZE)
        ax.set_ylim(0, self.SIZE)
        
        for i in range(self.NUM_PLAYERS):
            player_x = self.players[i].get_x()
            player_y = self.players[i].get_y()

            rect = plt.Circle((player_y, player_x), radius=0.5, facecolor="blue")  # 根据玩家的索引选择颜色
            ax.add_patch(rect)

        enemies = set()
        for i in range(self.NUM_ENEMIES):
            enemies.add((self.enemies[i].get_x(), self.enemies[i].get_y()))

        for enemy in enemies:  # 绘制敌人-红色
            rect = plt.Rectangle((enemy[1], enemy[0]), 1, 1, facecolor='red')
            ax.add_patch(rect)


        food = (self.food.get_x(), self.food.get_x())
        rect = plt.Circle((food[1], food[0]), radius=0.5, facecolor='green')  # 绘制食物-绿色
        ax.add_patch(rect)

        # 绘制智能体轨迹
        colors = ['yellow', 'orange', 'pink', 'black']  # 定义不同轨迹的颜色

        for i in range(len(self.trajectory)):
            x = [point[1] for point in self.trajectory[i]]
            y = [point[0] for point in self.trajectory[i]]
            ax.plot(x, y, color=colors[i], linewidth=5)

        if flag==1:
            plt.savefig("/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory_1.png")  # 保存图像到文件
        if flag==2:
            plt.savefig("trajectory_2.png")  # 保存图像到文件
        if flag==3:
            plt.savefig("trajectory_3.png")  # 保存图像到文件
        plt.close(fig)
        
