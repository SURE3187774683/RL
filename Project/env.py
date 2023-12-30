# grid的地图中agent躲避enemy，想吃到food
# agent、enemy的数量可自定义
# 静态和动态环境可选择
# 奖惩机制可自定义
# SURE
import pickle
import cv2
from PIL import Image
import numpy as np

ENV_MOVE = False                            #env是否变化
MAX_STEP = 200                              #每局最大步数

# 建立Cube类，用于创建player、food和enemy
class Cube:
    def __init__(self, size):  # 随机生成个体的位置
        self.size = size
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
    SIZE = 10         #地图大小
    NUM_PLAYERS = 1   # player的数量
    NUM_ENEMIES = 1   # enemy的数量

    OBSERVATION_SPACE_VALUES = (2+2*NUM_ENEMIES)*NUM_PLAYERS  # state的数量
    ACTION_SPACE_VALUES = 4 #action的数量

    FOOD_REWARD = 100
    ENEMY_PENALITY = -10
    MOVE_PENALITY = -0.1
    CLOSER_REWARD = 0.2
    FARER_PENALITY = -0.2
    STAY_REWARD = 0.1

    # 创建一个字典，用于存放agent的RGB
    d = {1: (255, 0, 0),  # blue
         2: (0, 255, 0),  # green
         3: (0, 0, 255)}  # red
    
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3

    def reset(self):
        self.trajectory = []          # 在每个步骤开始之前清空轨迹列表
        
        self.old_distances = 0
        self.players = []            # 创建players列表
        for i in range(self.NUM_PLAYERS):
            self.player = Cube(self.SIZE)        # 创建player
            self.players.append(self.player)

        self.food = Cube(self.SIZE)         # 创建food
        for i in range(self.NUM_PLAYERS):   
            while self.food == self.players[i]:
                self.food = Cube(self.SIZE)

        self.enemies = []                   # 创建enemy
        for i in range(self.NUM_PLAYERS):
            for j in range(self.NUM_ENEMIES):
                self.enemy = Cube(self.SIZE)
                while self.enemy == self.players[i] or self.enemy == self.food or self.enemy in self.enemies:
                    self.enemy = Cube(self.SIZE)
                self.enemies.append(self.enemy)

        state = ()
        for i in range(self.NUM_PLAYERS):
            state += (self.players[i] - self.food)
            for j in range(self.NUM_ENEMIES):
                state += (self.players[i] - self.enemies[j])
        self.episode_step = 0

        return state

    def step(self, action):
        equal_p_e = False
        self.episode_step += 1        

        for i in range(self.NUM_PLAYERS):
            self.players[i].action(action)

        if ENV_MOVE == True:
            self.food.move()
            for enemy in self.enemies:
                enemy.move()
   
        new_observation = () 
        new_distances = [] #下一步的每个agent和food的距离之和
        for i in range(self.NUM_PLAYERS):
            new_observation += (self.players[i] - self.food)
                # 考虑和每个enemy的位置关系
            new_distances = np.linalg.norm(new_observation, ord=1)     #计算agent和food的距离
            for j in range(self.NUM_ENEMIES):
                new_observation += (self.players[i] - self.enemies[j])
            
        # 判断player和enemy是否重叠
        if self.old_distances>new_distances:
            reward = self.CLOSER_REWARD
        elif self.old_distances<new_distances:
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
            self.trajectory.append((self.players[i].get_x(), self.players[i].get_y()))

        #所有玩家被吃掉/都到达/超过200步，游戏结束
        for i in range(self.NUM_PLAYERS):
            for j in range(self.NUM_ENEMIES):
                if self.players[i] == self.food or self.players[i] == self.enemies[j] or self.episode_step >= MAX_STEP:
                    done = True

        return new_observation, reward, done
        
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.get_x()][self.food.get_y()] = self.d[self.FOOD_N]

        for i in range(self.NUM_PLAYERS):        
            env[self.players[i].get_x()][self.players[i].get_y()] = self.d[self.PLAYER_N]

        for i in range(self.NUM_ENEMIES):
            env[self.enemies[i].get_x()][self.enemies[i].get_y()] = self.d[self.ENEMY_N]

        img = Image.fromarray(env, 'RGB')
        return img

    def render_trajectory(self,flag):   #收集agent的路径轨迹点
        img = Image.new('RGB', (self.SIZE, self.SIZE), (0, 0, 0))  # 创建一个空白的RGB图像
        
        agent = (self.players[0].get_x(),self.players[0].get_y())
        food = (self.food.get_x(), self.food.get_x())

        enemies = set()
        for i in range(self.NUM_ENEMIES):  # 使用range创建范围对象
            enemies.add((self.enemies[i].get_x(), self.enemies[i].get_y()))

        for enemy in enemies:                   # 绘制敌人-红色
            img.putpixel((enemy[1], enemy[0]),  (255, 0, 0))

        #img.putpixel((agent[1], agent[0]),(0, 0, 255))   # 绘制智能体-蓝色
        #img.putpixel((food[1], food[0]), (0, 255, 0))   # 绘制食物-绿色

        img_arr = np.array(img)  # 将PIL图像转换为NumPy数组
        
        # 绘制智能体轨迹
        for i in range(len(self.trajectory) - 1):
            point1 = ((self.trajectory[i][0]), (self.trajectory[i][1]))
            point2 = ((self.trajectory[i+1][0]), (self.trajectory[i+1][1]))
            cv2.line(img_arr, point1, point2, (255, 255, 255), 1)

        img = Image.fromarray(img_arr)  # 将NumPy数组转换为PIL图像
        img = img.resize((800, 800))
        img.show()

        if flag==1:
            img.save("trajectory_1.png")  # 保存带有轨迹的图像
        if flag==2:
            img.save("trajectory_2.png")  # 保存带有轨迹的图像

    def render(self):                   #显示图片
        img = self.get_image()
        img = img.resize((800, 800))
        cv2.imshow('Predator', np.array(img))
        cv2.waitKey(1)

# 用于qlearning
    def get_qtable(self, qtable_name=None): #搭建q table表格
        if qtable_name is None:
            q_table = {}

            def initialize_q_table(dimensions, ACTION_SPACE_VALUES):##定义q_table初始化函数
                q_table = {}
                recursive_initialize_q_table(
                    dimensions, ACTION_SPACE_VALUES, [], q_table)
                return q_table

            def recursive_initialize_q_table(dimensions, ACTION_SPACE_VALUES, indices, q_table):
                if len(indices) == len(dimensions):
                    q_table[tuple(indices)] = [np.random.uniform(-5, 0)
                                               for _ in range(ACTION_SPACE_VALUES)]
                else:
                    for i in range(-dimensions[len(indices)] + 1, dimensions[len(indices)]):
                        indices.append(i)
                        recursive_initialize_q_table(
                            dimensions, self.ACTION_SPACE_VALUES, indices, q_table)
                        indices.pop()

            dimensions = self.NUM_PLAYERS*(self.NUM_ENEMIES*2*[self.SIZE]+[self.SIZE]+[self.SIZE])
            q_table = initialize_q_table(dimensions, self.ACTION_SPACE_VALUES)     
        else:
            with open(qtable_name, 'rb') as f:
                q_table = pickle.load(f)
        return q_table
