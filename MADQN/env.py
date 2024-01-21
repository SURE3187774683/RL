# grid的地图中agent躲避enemy，想吃到food
# agent、enemy的数量可自定义
# 静态和动态环境可选择
# 奖惩机制可自定义
# SURE

import numpy as np
import matplotlib.pyplot as plt

# 建立Cube类，用于创建agent、food和enemy
class Cube:
    def __init__(self, size, x=None, y=None):  # 生成个体的位置
        self.size = size
        if x is not None and y is not None:
            self.x = x
            self.y = y
        else:
            self.x = np.random.randint(0, self.size)
            self.y = np.random.randint(0, self.size)

    def get_position(self):
        return (self.x, self.y)

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
    def __init__(self) -> None:
        self.SIZE = 30         #地图大小
        self.ENV_MOVE = False                            #env是否变化
        self.MAX_STEP = 300                              #每局最大步数
        self.ACTION_SPACE_VALUES = 9
        
        #奖励机制
        self.FOOD_REWARD = 100
        self.ENEMY_PENALITY = -10
        self.MOVE_PENALITY = -1
        self.CLOSER_REWARD = 10
        self.FARER_PENALITY = -10
        self.STAY_REWARD = 0

        #agent和enemy位置
        self.agent_positions = [(0,15),(0,0)]
        self.enemy_positions  = [(5,8),(6,8), 
                                 (5,9),(6,9),
                                 (5,10),(6,10),
         
                                 (9,2), (10,2),(11,2),(12,2),(13,2), (14,2),(15,2),(16,2),(17,2),(9,3),(9,4),
                                 
                                 (22,5), (23,5),(24,5),
                                 (22,6), (23,6),(24,6),
                                 (22,7), (23,7),(24,7), 
                                 (22,8), (23,8),(24,8),
                                 (22,9), (23,9),(24,9),
         
                                 (11,10),(12,10),(13,10),(14,10),(15,10),(16,10),
                                 (11,11),(12,11),
                                 (11,12),(12,12),
                                 (11,13),(12,13),
                                 
                                 (3,20),(4,20),(5,20),
                                 (3,21),(4,21),(5,21),(3,23),(3,24),(3,25),
                                 (3,22),(4,22),(5,22),
         
                                 (14,20), (14,21),(14,22),(14,23), (14,24),(14,25),(14,26),(14,27),
                                                  (15,22),(15,23), (15,24),(15,25),(15,26),(15,27),
                                 
                                 (21,17), (22,17),(23,17),(24,17), (25,17),(26,17),
                                 (21,18), (22,18),(23,18),(24,18), (25,18),(26,18),(26,19),
                                 (26,20),(26,21),(26,22),(26,23),(26,24)]  # 指定enemy的位置
        self.NUM_PLAYERS = len(self.agent_positions)    # agent的数量
        self.NUM_ENEMIES = len(self.enemy_positions)     # enemy的数量
        self.OBSERVATION_SPACE_VALUES = (2+2*self.NUM_ENEMIES)*self.NUM_PLAYERS  # state的数量
        
    def reset(self):
        self.trajectory = [[] for _ in range(self.NUM_PLAYERS)]          # 在每个步骤开始之前清空轨迹列表

        self.agent_dead = [0]*self.NUM_PLAYERS
        self.old_distances = float('inf')  # 初始化为正无穷大

        # 创建food,agents,enemy
        self.food = Cube(self.SIZE,23,23)       
        self.agents = []                       
        for i in range(self.NUM_PLAYERS):
            x, y = self.agent_positions[i]  
            self.agent = Cube(self.SIZE, x, y)  
            self.agents.append(self.agent)
        self.enemies = []                       
        for i in range(self.NUM_ENEMIES):
            x, y = self.enemy_positions[i]  
            self.enemy = Cube(self.SIZE, x, y)  
            self.enemies.append(self.enemy)

        state = ()                              # 记录状态
        for i in range(self.NUM_PLAYERS):
            state += (self.agents[i] - self.food)
            for j in range(self.NUM_ENEMIES):
                state += (self.agents[i] - self.enemies[j])
        self.episode_step = 0

        return state

    def step(self, agent_id,action):
        equal_p_e = False
        self.episode_step += 1
        
        # 当agent遇到enemy时将该agent的移动设置为静止
        if self.agents[agent_id] in self.enemies:
            self.agents[agent_id].action(8)
            self.agent_dead[agent_id] = 1

        if not self.agent_dead[agent_id]:
            self.agents[agent_id].action(action)      

        if self.ENV_MOVE == True:
            self.food.move()
            for enemy in self.enemies:
                enemy.move()
   
        new_observation = () 

        new_distances = [] #下一步的每个agent和food的距离之和
        for i in range(self.NUM_PLAYERS):
            new_observation += (self.agents[i] - self.food)    # 更新state
            distance = np.linalg.norm(new_observation, ord=1)   # 计算代理和食物的距离
            new_distances.append(distance)
            
            for j in range(self.NUM_ENEMIES):
                new_observation += (self.agents[i] - self.enemies[j])

        new_distances_sum = np.sum(new_distances)

        if self.old_distances>new_distances_sum:
            reward = self.CLOSER_REWARD
        elif self.old_distances<new_distances_sum:
            reward = self.FARER_PENALITY
        else:
            reward = self.STAY_REWARD
        
        self.old_distances = new_distances_sum

        for i in range(self.NUM_PLAYERS):           #定义奖励机制
            for j in range(self.NUM_ENEMIES):
                if self.agents[i] == self.enemies[j]:
                    equal_p_e = True

            if self.agents[i] == self.food:
                reward += self.FOOD_REWARD
            elif equal_p_e:
                reward += self.ENEMY_PENALITY
            else:
                reward += self.MOVE_PENALITY
        
        #游戏结束标志
        done = False
        #任意一个玩家到达food，游戏结束
        for i in range(self.NUM_PLAYERS):
            if self.agents[i] == self.food :
                done = True

        #当步数大于200,游戏结束
        if self.episode_step >= self.MAX_STEP:
            done = True

        #任意两个玩家相撞，游戏结束
        for i in range(self.NUM_PLAYERS):
            for j in range(i+1, self.NUM_PLAYERS):
                if self.agents[i].get_x() == self.agents[j].get_x() and self.agents[i].get_y() == self.agents[j].get_y():
                    done = True

        #所有agent都死了，游戏结束
        if all(element == 1 for element in self.agent_dead):
            self.agent_dead = [0]*self.NUM_PLAYERS
            done = True

        #将智能体的位置添加到轨迹列表中
        for i in range(self.NUM_PLAYERS):
            self.trajectory[i].append((self.agents[i].get_x(), self.agents[i].get_y()))

        return new_observation, reward, done
        
    def render_trajectory(self, flag):          #记录agent轨迹
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.SIZE)
        ax.set_ylim(0, self.SIZE)
        
        for i in range(self.NUM_PLAYERS):
            agent_x = self.agents[i].get_x()
            agent_y = self.agents[i].get_y()

            rect = plt.Circle((agent_y, agent_x), radius=0.5, facecolor="blue")  # 根据玩家的索引选择颜色
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
        colors = ['yellow', 'orange', 'pink', 'black','blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # 定义不同轨迹的颜色

        for i in range(len(self.trajectory)):
            x = [point[1] for point in self.trajectory[i]]
            y = [point[0] for point in self.trajectory[i]]
            ax.plot(x, y, color=colors[i], linewidth=5)

        if flag==1:
            plt.savefig("/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory.png")  # 保存图像到文件
        if flag==2:
            plt.savefig("trajectory_2.png")  # 保存图像到文件
        if flag==3:
            plt.savefig("/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory_3.png")  # 保存图像到文件
        plt.close(fig)
        