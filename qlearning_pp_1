##原始版本：一个enemy



import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

EPISODES = 30000  # 训练的回合数
SHOW_EVERY = 3000  # 每隔几次展示一次数据

epsilon = 0.6  # 随机选择action的概率
EPS_DECAY = 0.9998  # 随机选择的衰减率
DISCOUNT = 0.95  # i+1次state value的影响程度
LEARNING_RATE = 0.1  # 学习速率（步长）


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
    SIZE = 10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # state的数量
    ACTION_SPACE_VALUES = 9  # action的数量
    RETURN_IMAGE = False

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
        self.player = Cube(self.SIZE)  # 创建player
        self.food = Cube(self.SIZE)  # 创建food
        while self.food == self.player:
            self.food = Cube(self.SIZE)

        self.enemy = Cube(self.SIZE)  # 创建enemy
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Cube(self.SIZE)

        if self.RETURN_IMAGE:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food)+(self.player - self.enemy)

        self.episode_step = 0

        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)
        self.food.move()
        self.enemy.move()

        if self.RETURN_IMAGE:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + \
                (self.player - self.enemy)

        if self.player == self.food:
            reward = self.FOOD_REWARD
        elif self.player == self.enemy:
            reward = self.ENEMY_PENALITY
        else:
            reward = self.MOVE_PENALITY

        done = False
        if self.player == self.food or self.player == self.enemy or self.episode_step >= 200:
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
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        img = Image.fromarray(env, 'RGB')
        return img

    def get_qtable(self, qtable_name=None):
        if qtable_name is None:
            q_table = {}
            for x1 in range(-self.SIZE+1, self.SIZE):
                for y1 in range(-self.SIZE+1, self.SIZE):
                    for x2 in range(-self.SIZE+1, self.SIZE):
                        for y2 in range(-self.SIZE+1, self.SIZE):
                            q_table[(x1, y1, x2, y2)] = [np.random.uniform(-5, 0)
                                                         for i in range(self.ACTION_SPACE_VALUES)]
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
        show = True
    else:
        show = False

    episode_reward = 0
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, env.ACTION_SPACE_VALUES)

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

# with open(f'qtable_{int(time.time())}.pickle','wb') as f:
# pickle.dump(q_table,f)
