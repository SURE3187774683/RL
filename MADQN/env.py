# grid的地图中agent躲避enemy，想吃到food
# agent、enemy的数量可自定义
# 静态和动态环境可选择
# 奖惩机制可自定义
# SURE
import json
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
        self.MAX_STEP = 200                              #每局最大步数
        self.ACTION_SPACE_VALUES = 9
        
        #奖励机制
        self.rewards = {
            'find_food': 100,
            'meet_enemy': -20,
            'move': -1,
            'closer': 10,
            'farer': -10,
            'stay': 0
        }

        #轨迹保存路径
        self.flag_file = {
           1: "/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory.png", 
           2: "/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory_2.png",
           3: "/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory_3.png",
           4: "/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory_4.png", 
           5: "/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory_5.png",
           6: "/mnt/c/Users/asus/Desktop/trajectory_picture/trajectory_6.png"
        }

        #获取agent和enemy位置
        with open('positions.json', 'r') as f:
            positions = json.load(f)
        self.agent_positions = [tuple(pos) for pos in positions['agent_positions']]
        self.enemy_positions = [tuple(pos) for pos in positions['enemy_positions']]

        self.NUM_PLAYERS = len(self.agent_positions)    # agent的数量
        self.NUM_ENEMIES = len(self.enemy_positions)     # enemy的数量
        self.OBSERVATION_SPACE_VALUES = (2+2*self.NUM_ENEMIES)*self.NUM_PLAYERS  # state的数量
        
    def reset(self):
        self.agents_rewards = [0] * self.NUM_PLAYERS  # 重置每个 agent 的奖励为 0
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
        
        # 记录状态
        state = []                            
        for i in range(self.NUM_PLAYERS):
            state += (self.agents[i] - self.food)
            for j in range(self.NUM_ENEMIES):
                state += (self.agents[i] - self.enemies[j])
        self.episode_step = 0

        self.trajectory = [[] for _ in range(self.NUM_PLAYERS)]          # 在每个步骤开始之前清空轨迹列表

        self.agent_dead = [0]*self.NUM_PLAYERS  # agent复活
        self.agent_find = [0]*self.NUM_PLAYERS  # agent均未找到food

        self.old_distances = float('inf')  # 初始化为正无穷大

        return state

    def step(self, agent_id, action):
        self.episode_step += 1

        # 设置环境的移动和静止
        if self.ENV_MOVE == True:
            self.food.move()
            for enemy in self.enemies:
                enemy.move()
        
        # 当agent遇到enemy时或者遇到food时，将该agent的移动设置为静止
        if self.agents[agent_id] == self.food or self.agents[agent_id] in self.enemies:
            self.agents[agent_id].action(8)
            self.agent_find[agent_id] = 1
            self.agent_dead[agent_id] = 1

        # 当agent没有遇到enemy并且没有找到food时继续action
        if not self.agent_dead[agent_id] and not self.agent_find[agent_id]:
            self.agents[agent_id].action(action)      

        # 更新state
        new_observation = [] 
        for i in range(self.NUM_PLAYERS):
            new_observation += (self.agents[i] - self.food)   
            for j in range(self.NUM_ENEMIES):
                new_observation += (self.agents[i] - self.enemies[j])
        
        # 计算代理和食物的距离
        new_distances = [] 
        for i in range(self.NUM_PLAYERS):
            distance = np.linalg.norm(new_observation, ord=1)   
            new_distances.append(distance)
        new_distances_sum = np.sum(new_distances)

        # 奖励机制一：基于agent和food间的距离
        if self.old_distances>new_distances_sum:
            reward = self.rewards['closer']
        elif self.old_distances<new_distances_sum:
            reward = self.rewards['farer']
        else:
            reward = self.rewards['stay']
        
        self.old_distances = new_distances_sum

        # 奖励机制二：只有当 agent 没有遇到 enemy 或者 food 时，才会更新奖励
        if not self.agent_dead[agent_id] and not self.agent_find[agent_id]:
            self.agents_rewards[agent_id] += reward
 
        # 当 agent 遇到 enemy 或者 food 时，只需要将 reward 加上相应的奖惩即可
        if self.agents[agent_id] == self.food:
            reward += self.rewards['find_food']
        elif self.agents[agent_id] in self.enemies:
            reward += self.rewards['meet_enemy']
        else:
            reward += self.rewards['move']
        
        #游戏结束标志
        done = False

        #任意两个玩家相撞，游戏结束
        for i in range(self.NUM_PLAYERS):
            for j in range(i+1, self.NUM_PLAYERS):
                if self.agents[i].get_x() == self.agents[j].get_x() and self.agents[i].get_y() == self.agents[j].get_y():
                    done = True

        #所有agent都遇到enemy或者都找到food了，游戏结束
        if all(element == 1 for element in self.agent_dead) or all(element == 1 for element in self.agent_find):
            done = True

        #当步数大于200,游戏结束
        if self.episode_step >= self.MAX_STEP:
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
        
        plt.savefig(self.flag_file[flag])  # 保存图像到文件
        plt.close(fig)