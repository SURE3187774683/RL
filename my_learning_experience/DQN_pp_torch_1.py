# 第一版：用pytorch代替q_table,one agent,one enemy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

##########################################################################
memory_capacity = 50000     #经验池的大小
batch_size = 30             #每次从经验池中取出的个数
gamma = 0.99                #学习率

epsilon_start = 1.0         #epsilon的初始值
epsilon_end = 0.01          #epsilon的终止值
epsilon_decay = 0.9995       #epsilon的缩减速率

num_steps=10000              #总训练次数
visualize=False             #是否观看回放
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
    SIZE = 10
    
    NUM_PLAYERS = 1  # player的数量
    NUM_ENEMIES = 1  # enemy的数量

    OBSERVATION_SPACE_VALUES = (4,)  # state的数量
    ACTION_SPACE_VALUES = 4 #action的数量
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
    

########################################################################
   
class Model(nn.Module):
    def __init__(self, input_shape, output_shape):      #生成一个state数量输入，action数量输出的神经网络
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape[0], 32)
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

def build_model(input_shape, output_shape):
    model = Model(input_shape, output_shape)
    return model


class DQN(nn.Module):       #定义了深度神经网络模型,用于构建当前q_net和目标q_net
    def __init__(self, input_shape, nb_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, nb_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:     #经验回放缓存
    def __init__(self, capacity):   #随机生成capacity大小的经验池
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):    #将经验存储到缓存中
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):                               #从缓存中随机采样一批经验
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):      #当前缓存中的经验数量
        return len(self.memory)

class DQNAgent:
    average_rewards = []  # 用于保存每个回合的平均奖励
    def __init__(self, nb_states, nb_actions, memory_capacity, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):       #生成agent的参数

        self.nb_states = nb_states
        self.nb_actions = nb_actions  
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory = ReplayMemory(memory_capacity)     #随机生成50000容量的经验池
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(nb_states, nb_actions).to(self.device)    #构建当前q_net
        self.target_net = DQN(nb_states, nb_actions).to(self.device)    #构建目标q_net
        self.target_net.load_state_dict(self.policy_net.state_dict())   #使两个q_net结构相同
        self.target_net.eval()                                          #将目标网络设为评估模式
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)  #建立优化器来更新策略网络的权重
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
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def push_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update_model(self):     #将采样的经验转换为 PyTorch 张量
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (~dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_epsilon()
        
    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def train(self, env, num_steps, visualize, verbose):    #训练agent
        state = env.reset()
        episode_reward = 0      #记录当次训练的reward
        
        total_reward = 0        #记录总训练次数
        episode_count = 0       # 新增的变量用于保存游戏的总数
        
        for step in range(num_steps):
            action = self.select_action(state)              #选择action
            next_state, reward, done = env.step(action)     #游戏走一步
            episode_reward += reward                        #累加当次训练的reward
            total_reward += reward

            self.push_transition(state, action, reward, next_state, done)   #将当前状态放入经验池
            self.update_model()
            
            if done:
                state = env.reset()
                episode_count += 1

                if verbose == 1:        #当verbose为1时，输出游戏回合数和平均奖励
                    average_reward = total_reward / episode_count
                    DQNAgent.average_rewards.append(average_reward)
                    print(f"Step: {step}, Epsilon:{self.epsilon}")
                    print(f"### Average Reward: {average_reward}")

                if verbose == 2:        #当verbose为2时，输出游戏回合数和每轮奖励
                    print(f"Step: {step}, Episode Reward: {episode_reward}")
                episode_reward = 0
            
                
            else:
                state = next_state
                
            if step % 100 == 0:
                self.update_target_model()
                
            if visualize:
                env.render()

def show_table(if_show):        #是否要展示episode和average_reward的关系
    if if_show==True:
        # 定义卷积核
        window_size = 10
        kernel = np.ones(window_size) / window_size

        # 对平均奖励数据进行卷积操作
        smoothed_rewards = np.convolve(DQNAgent.average_rewards, kernel, mode='valid')

        # 绘制平滑后的曲线
        plt.plot(range(window_size, window_size + len(smoothed_rewards)), smoothed_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Smoothed Average Reward per Episode')
        plt.show()



###############################################################################################################
env = envCube()
model = build_model(env.OBSERVATION_SPACE_VALUES,env.ACTION_SPACE_VALUES)   #建立以state数量为输入，action数量为输出的神经网络
agent = DQNAgent(env.OBSERVATION_SPACE_VALUES[0], env.ACTION_SPACE_VALUES, memory_capacity, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay)
agent.train(env, num_steps, visualize, verbose=1)        #训练agent
show_table(True)
#图表展示训练效果


