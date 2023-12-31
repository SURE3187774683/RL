# 第五版：多个enemy，多个player，qtable每个player有关

import numpy as np
import cv2
from PIL import Image
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


q_table = None     #是否调用已知数据
EPISODES = 30000  # 训练的回合数
SHOW_EVERY = 3000  # 每隔几次展示一次数据

epsilon = 0.5  # 随机选择action的概率
EPS_DECAY = 0.9998  # 随机选择的衰减率
DISCOUNT = 0.95  # i+1次state value的影响程度
LEARNING_RATE = 0.1  # 学习速率（步长）
ACTION_SPACE_VALUES = 4 #action的数量

# 建立Cube类，用于创建player、food和enemy


class Cube:
    def __init__(self, size):  # 随机生成个体的位置
        self.size = size
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)

    def __str__(self):  # 将位置转化为字符串形式
        return f'{self.x},{self.y}'

    def __sub__(self, other):  # 位置相减（subtraction）
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):  # 判断两个智能体位置是否相同
        return self.x == other.x and self.y == other.y

    def action(self, choise):  # action函数（向8个位置移动或静止）
        if choise == 0:
            self.move(x=1, y=1)
        elif choise == 1:
            self.move(x=-1, y=1)
        elif choise == 2:
            self.move(x=1, y=-1)
        elif choise == 3:
            self.move(x=-1, y=-1)
        elif choise == 4:
            self.move(x=0, y=1)
        elif choise == 5:
            self.move(x=0, y=-1)
        elif choise == 6:
            self.move(x=1, y=0)
        elif choise == 7:
            self.move(x=-1, y=0)
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
    SIZE = 4
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # state的数量
    RETURN_IMAGE = False
    NUM_ENEMIES = 1  # enemy的数量
    NUM_PLAYERS = 2  # player的数量

    FOOD_REWARD = 25
    ENEMY_PENALITY = -300
    MOVE_PENALITY = -1

    # 创建一个字典，用于存放agent的RGB
    d = {1: (255, 0, 0),  # blue
         2: (0, 255, 0),  # green
         3: (0, 0, 255)}  # red

    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3

    def reset(self):
        self.players = []            # 创建players列表
        for i in range(self.NUM_PLAYERS):
            player = Cube(self.SIZE)    # 创建player
            self.players.append(player)

        self.food = Cube(self.SIZE)  # 创建food
        for i in range(self.NUM_PLAYERS):
            while self.food == self.players[i]:
                self.food = Cube(self.SIZE)

        self.enemies = []            # 创建敌人列表
        for i in range(self.NUM_PLAYERS):
            for j in range(self.NUM_ENEMIES):
                enemy = Cube(self.SIZE)    # 创建敌人
                while enemy == self.players[i] or enemy == self.food or enemy in self.enemies:
                    enemy = Cube(self.SIZE)
                self.enemies.append(enemy)

        if self.RETURN_IMAGE:
            observation = np.array(self.get_image())
        else:
                observation = ()
                for i in range(self.NUM_PLAYERS):
                    observation += (self.players[i] - self.food)
                    # 考虑和每个enemy的位置关系
                    for j in range(self.NUM_ENEMIES):
                        observation += (self.players[i] - self.enemies[j])
        self.episode_step = 0

        return observation

    def step(self, action):
        self.episode_step += 1   
        for i in range(self.NUM_PLAYERS):
            self.players[i].action(action)
        # self.food.move()
        # for enemy in self.enemies:
        #    enemy.move()

        if self.RETURN_IMAGE:
            new_observation = np.array(self.get_image())
        else:   
            new_observation = ()      
            for i in range(self.NUM_PLAYERS):
                new_observation += (self.players[i] - self.food)
                    # 考虑和每个enemy的位置关系
                for j in range(self.NUM_ENEMIES):
                    new_observation += (self.players[i] - self.enemies[j])
# 判断player和enemy是否重叠
        equal_p_e = False
        for i in range(self.NUM_PLAYERS):        
            for j in range(self.NUM_ENEMIES):
                if self.players[i] == self.enemies[j]:
                    equal_p_e = True
            if self.players[i] == self.food:
                reward = self.FOOD_REWARD

            elif equal_p_e:
                reward = self.ENEMY_PENALITY

            else:
                reward = self.MOVE_PENALITY
        done = False
        #所有玩家被吃掉/都到达/超过200步，游戏结束
        for i in range(self.NUM_PLAYERS):
            for j in range(self.NUM_ENEMIES):
                if self.players[i] == self.food or self.players[i] == self.enemies[j] or self.episode_step >= 200:
                    done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((800, 800))
        cv2.imshow('Predator', np.array(img))
        cv2.waitKey(1)

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]

        for i in range(self.NUM_PLAYERS):        
            env[self.players[i].x][self.players[i].y] = self.d[self.PLAYER_N]

        for i in range(self.NUM_ENEMIES):
            env[self.enemies[i].x][self.enemies[i].y] = self.d[self.ENEMY_N]

        img = Image.fromarray(env, 'RGB')
        return img

    def get_qtable(self, qtable_name=None):
        if qtable_name is None:
            q_table = {}

##############################################################################################
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
                            dimensions, ACTION_SPACE_VALUES, indices, q_table)
                        indices.pop()

            dimensions = self.NUM_PLAYERS*(self.NUM_ENEMIES*2*[self.SIZE]+[self.SIZE]+[self.SIZE])
            q_table = initialize_q_table(dimensions, ACTION_SPACE_VALUES)
###############################################################################################        
        else:
            with open(qtable_name, 'rb') as f:
                q_table = pickle.load(f)
        return q_table


# 创建环境和智能体
env = envCube()
q_table = env.get_qtable()

episode_rewards = []
for episode in range(EPISODES):
    obs = env.reset()
    done = False

    if episode % SHOW_EVERY == 0:
        print(f'episode #{episode}, epsilon:{epsilon}')
        print(f'mean reward:{np.mean(episode_rewards[-SHOW_EVERY:])}')
        #print(f'q_table:{q_table}')

        show = True
    else:
        show = False

    episode_reward = 0
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, ACTION_SPACE_VALUES)

        new_obs, reward, done = env.step(action)

        # Update the Q_table
        current_q = q_table[obs][action]
        max_future_q = np.max(q_table[new_obs])
        if reward == env.FOOD_REWARD:
            new_q = env.FOOD_REWARD
        else:
            new_q = current_q + LEARNING_RATE * \
                (reward+DISCOUNT*max_future_q-current_q)
        q_table[obs][action] = new_q
        obs = new_obs

        if show:
            env.render()

        episode_reward += reward

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones(
    (SHOW_EVERY,))/SHOW_EVERY, mode='valid')
print(len(moving_avg))
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.xlabel('episode #')
plt.ylabel(f'mean {SHOW_EVERY} reward')
plt.show()

#with open(f'qtable_{int(time.time())}.pickle','wb') as f:
#    pickle.dump(q_table,f)
