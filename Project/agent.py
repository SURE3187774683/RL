#创建agent

import torch
import torch.nn as nn
import random
from collections import deque
import torch.nn as nn
import torch.optim as optim
import random

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
    
    def sample(self, BATCH_SIZE):                               #从缓存中随机采样一批经验并将其转换为张量形式
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = random.sample(self.memory, BATCH_SIZE)          
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):      #当前缓存中的经验数量
        return len(self.memory)

class DQNAgent:
    episode_rewards = []
    loss_value = 0                                 #每局loss清零
    def __init__(self, nb_states, nb_actions, REPLAY_MEMORY_SIZE, BATCH_SIZE, DISCOUNT, LEARNING_RATE,EPI_START, EPI_END, epsilon_decay):       #生成agent的参数
        self.replay_memory_size = REPLAY_MEMORY_SIZE
        self.learning_rate = LEARNING_RATE
        self.nb_states = nb_states
        self.nb_actions = nb_actions  
        self.BATCH_SIZE = BATCH_SIZE
        self.discount = DISCOUNT
        self.epsilon = EPI_START
        self.EPI_END = EPI_END
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(nb_states, nb_actions).to(self.device)    #构建当前q_net
        self.target_net = DQN(nb_states, nb_actions).to(self.device)    #构建目标q_net
        self.target_net.load_state_dict(self.policy_net.state_dict())   #使两个q_net结构相同
        self.target_net.eval()                                          #将目标网络设为评估模式
        self.memory = ReplayMemory(self.replay_memory_size)                  #随机生成50000容量的经验池
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.learning_rate)  #建立优化器来更新策略网络的权重
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
    
    def update_model(self):     #将采样的经验转换为 PyTorch 张量
        if len(self.memory) < self.replay_memory_size:                               #经验池小于一定量时
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)#从经验池取样
        q_values = self.policy_net(states).gather(1, actions)                   #链接action和对应的q_value
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)     #选取最大的value对应的q_value
        expected_q_values = rewards + (~dones) * self.discount * next_q_values     #贝尔曼方程计算期望的 Q 值 
        loss = self.loss_fn(q_values, expected_q_values)                        #当前策略网络的 Q 值估计与期望 Q 值之间的差异
        self.loss_value = loss.item()                                           #将损失值保存在loss_value变量中
        self.optimizer.zero_grad()                                              #将优化器的梯度缓冲区清零
        loss.backward()                                                         #通过自动求导计算损失函数关于网络参数的梯度
        self.optimizer.step()                                                   #根据计算得到的梯度，更新网络参数
        
    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        
