# 第五版：减去一个MODEL类之后的修正版,加入tensorboard
import torch
import time
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
from tensorboardX import SummaryWriter
import random
import numpy as np
import cv2
from PIL import Image

##########################################################################
EPISODE_N = 10000                           #总训练局数
REPLAY_MEMORY_SIZE = 100                    #经验池的大小
BATCH_SIZE = 32                             #每次从经验池中取出的个数
gamma = 0.95                                #折扣因子
lr = 1e-3                                   #学习率(步长)
UPDATE_TARGET_MODE_EVERY = 20               #model更新频率
STATISTICS_EVERY = 5                        #记录在tensorboard的频率

model_save_avg_reward = 80                  #评价指标
JUDGE_REWARD = 80
EPI_START = 1                               #epsilon的初始值
EPI_END = 0.001                             #epsilon的终止值
EPI_DECAY = 0.999995                        #epsilon的缩减速率
#########################################################################
VISUALIZE = False                           #是否观看回放
ENV_MOVE = False                            #env是否变化
VERBOSE = 1                                 #调整日志模式（1——平均游戏得分；2——每局游戏得分）
MAX_STEP = 200                              #每局最大步数
SHOW_EVERY = 100                              #显示频率
##########################################################################

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
    SIZE = 10           #地图大小
    NUM_PLAYERS = 1     # player的数量
    NUM_ENEMIES = 5   # enemy的数量

    OBSERVATION_SPACE_VALUES = (2+2*NUM_ENEMIES)*NUM_PLAYERS  # state的数量
    ACTION_SPACE_VALUES = 9 #action的数量

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
        equal_p_e = False

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

        #所有玩家被吃掉/都到达/超过200步，游戏结束
        for i in range(self.NUM_PLAYERS):
            for j in range(self.NUM_ENEMIES):
                if self.players[i] == self.food or self.players[i] == self.enemies[j] or self.episode_step >= MAX_STEP:
                    done = True

        return new_observation, reward, done
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]

        for i in range(self.NUM_PLAYERS):        
            env[self.players[i].x][self.players[i].y] = self.d[self.PLAYER_N]

        for i in range(self.NUM_ENEMIES):
            env[self.enemies[i].x][self.enemies[i].y] = self.d[self.ENEMY_N]

        img = Image.fromarray(env, 'RGB')
        return img

    def render(self):
        img = self.get_image()
        img = img.resize((800, 800))
        cv2.imshow('Predator', np.array(img))
        cv2.waitKey(1)

class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):      #生成一个state数量输入，action数量输出的神经网络
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, output_shape)
        

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class ReplayMemory:     #经验回放缓存
    def __init__(self, capacity):   #随机生成capacity大小的经验池
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):    #将经验存储到缓存中
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, BATCH_SIZE):                               #从缓存中随机采样一批经验
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):      #当前缓存中的经验数量
        return len(self.memory)

class DQNAgent:
    episode_rewards = []
    loss_value = 0
    losses = []  # 用于保存每一步的损失值

    def __init__(self, nb_states, nb_actions, REPLAY_MEMORY_SIZE, BATCH_SIZE, gamma, EPI_START, EPI_END, epsilon_decay):       #生成agent的参数
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(nb_states, nb_actions).to(self.device)    #构建当前q_net
        self.target_net = DQN(nb_states, nb_actions).to(self.device)    #构建目标q_net
        self.target_net.load_state_dict(self.policy_net.state_dict())   #使两个q_net结构相同
        self.target_net.eval()                                          #将目标网络设为评估模式

        self.nb_states = nb_states
        self.nb_actions = nb_actions  
        self.BATCH_SIZE = BATCH_SIZE
        self.gamma = gamma
        self.epsilon = EPI_START
        self.EPI_END = EPI_END
        self.epsilon_decay = epsilon_decay

        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)     #随机生成50000容量的经验池
        

        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr)  #建立优化器来更新策略网络的权重
        self.loss_fn = nn.MSELoss()                                     #定义损失函数为均方误差损失函数
        
    def select_action(self, state):     
        if random.random() < self.epsilon:      #当随机值小于epsilon时随机选择action
            return random.randrange(self.nb_actions)
        else:                                   #当随机值大于epsilon时选择值最大的action
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
    
    def update_epsilon(self):               
        self.epsilon = max(self.EPI_END, self.epsilon * self.epsilon_decay)
        
    def push_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update_model(self):     #利用经验池更新神经网络模型
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (~dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, expected_q_values)
        self.loss_value = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def train(self, mkdirenv, visualize, verbose):          #训练agent
        writer = SummaryWriter('logs/')                     #将结果画在tensorboard上
        for episode in range(EPISODE_N):
            state = env.reset()                             #重置环境
            done = False
            episode_reward = 0                              #每局奖励清零

            while not done:
                action = self.select_action(state)              #选择action
                next_state, reward, done = env.step(action)     #游戏走一步
                self.push_transition(state, action, reward, next_state, done)   #将当前状态放入经验池       
                self.update_model()                             #更新model
                state = next_state                              #更新state
                episode_reward += reward                        #累加当次训练的reward
                if visualize and episode%SHOW_EVERY == 0:
                    env.render()

            self.losses.append(self.loss_value)                 #收集所有训练累计的loss
            self.episode_rewards.append(episode_reward)         #收集所有训练累计的reward
            
            self.update_epsilon()
            if episode % UPDATE_TARGET_MODE_EVERY == 0:         #更新target_model(将当前模型的参数复制到目标模型)
                self.update_target_model()

            if episode%SHOW_EVERY==0:                           #打印日志             
                if episode_reward>JUDGE_REWARD:
                    print("WIN!")
                if episode_reward<JUDGE_REWARD:
                    print("LOSE")

                print(f"Episode: {episode}        Epsilon:{self.epsilon}")

                if verbose == 1:                                #输出平均奖励
                    print(f"### Average Reward: {np.mean(self.episode_rewards)}")                
                if verbose == 2:                                #输出每轮游戏的奖励
                    print(f"### Episode Reward: {self.episode_rewards[-1]}")
            
            model_save_avg_reward = 60
            if episode % STATISTICS_EVERY == 0:
                avg_reward = sum(self.episode_rewards[-STATISTICS_EVERY:])/len(self.episode_rewards[-STATISTICS_EVERY:])
                max_reward = max(self.episode_rewards[-STATISTICS_EVERY:])
                min_reward = min(self.episode_rewards[-STATISTICS_EVERY:])
                #print(f'avg_reward:{avg_reward},max_reward:{max_reward},min_reward:{min_reward}')
                
                
                writer.add_scalar('Average Reward', avg_reward, episode)
                writer.add_scalar('Max Reward', max_reward, episode)
                writer.add_scalar('Min Reward', min_reward, episode)
                writer.add_scalar('Epsilon', self.epsilon, episode)
                writer.add_scalar('Loss', self.loss_value, episode)
                
                if avg_reward > model_save_avg_reward:
                    model_save_avg_reward = avg_reward
                    model_dir = './models'
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    model_path = os.path.join(model_dir, f'{avg_reward:7.3f}_{int(time.time())}.model')
                    torch.save(DQN(env.OBSERVATION_SPACE_VALUES,env.ACTION_SPACE_VALUES).state_dict(), model_path)

###############################################################################################################

env = envCube()
agent = DQNAgent(env.OBSERVATION_SPACE_VALUES, env.ACTION_SPACE_VALUES, REPLAY_MEMORY_SIZE, BATCH_SIZE, gamma, EPI_START, EPI_END, EPI_DECAY)
agent.train(env,VISUALIZE, VERBOSE) 
