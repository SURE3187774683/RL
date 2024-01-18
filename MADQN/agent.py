# 神经网络：2*线性层、3*隐藏层
# loss函数：均方误差损失函数(Mean Squared Error)
# 优化器: Adam(自适应地调整学习率)
# SURE

import torch
import torch.nn as nn
import random
from collections import deque
import torch.nn as nn
import torch.optim as optim

class ReplayMemory:     #经验回放缓存
    def __init__(self, capacity,device):   #随机生成capacity大小的经验池
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):    #将经验存储到缓存中
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, BATCH_SIZE):                               #从缓存中随机采样一批经验并将其转换为张量形式
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

class DQNAgent:
    episode_rewards = []
    loss_value = 0                                 #每局loss清零
    def __init__(self, player_id,nb_states, nb_actions, REPLAY_MEMORY_SIZE, BATCH_SIZE, DISCOUNT, LEARNING_RATE,EPI_START, EPI_END, epsilon_decay,device):       #生成agent的参数
        self.replay_memory_size = REPLAY_MEMORY_SIZE
        self.learning_rate = LEARNING_RATE
        self.nb_states = nb_states
        self.nb_actions = nb_actions  
        self.BATCH_SIZE = BATCH_SIZE
        self.discount = DISCOUNT
        self.epsilon = EPI_START
        self.episilon_end = EPI_END
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.player_id = player_id
        self.policy_net = DQN(nb_states, nb_actions).to(self.device)
        self.target_net = DQN(nb_states, nb_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(self.replay_memory_size, self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.nb_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                
                return torch.argmax(q_values).item()

    def update_epsilon(self):               
        self.epsilon = max(self.episilon_end, self.epsilon * self.epsilon_decay)
        
    def push_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update_model(self):     #将采样的经验转换为 PyTorch 张量
        if len(self.memory) < self.replay_memory_size:                          #经验池小于一定量时
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)#从经验池取样
        q_values = self.policy_net(states).gather(1, actions)                   #链接action和对应的q_value
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)     #选取最大的value对应的q_value
        expected_q_values = rewards + (~dones) * self.discount * next_q_values  #贝尔曼方程计算期望的 Q 值 
        loss = self.loss_fn(q_values, expected_q_values)                        #当前策略网络的 Q 值估计与期望 Q 值之间的差异
        self.loss_value = loss.item()                                           #将损失(tensor)的值保存在loss_value变量中
        self.optimizer.zero_grad()                                              #将优化器的梯度缓冲区清零
        loss.backward()                                                         #通过自动求导计算损失函数关于网络参数的梯度
        self.optimizer.step()                                                   #根据计算得到的梯度，更新网络参数
        
        
    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

