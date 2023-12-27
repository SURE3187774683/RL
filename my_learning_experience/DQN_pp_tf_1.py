#第一版 基于tf的单个agent路径规划，基于图像位置和cnn
from sys import setprofile
import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

from keras.models import Sequential             #构建神经网络的库函数
from keras.layers import Dense,Conv2D,Flatten   #导入全连接层、卷积层、平滑层     
from collections import deque                   #构建双向链表
import random
import os
import tensorflow as tf
from keras.callbacks import TensorBoard

##########################################################################
EPISODES = 10000                            #训练的总局数
REPLAY_MEMORY_SIZE = 2000                   #经验池的大小
MINI_REPLAY_MEMORY_SIZE = 100               #从经验池取出的transition个数
DISCOUNT = 0.95                             #折扣回报率
UPDATE_TARGET_MODE_EVERY = 20               #model更新频率
STATISTICS_EVERY = 5                        #记录在tensorboard的频率
MAX_STEP = 200                              #每局最大步数
SHOW_EVERY = 10                             #render的显示频率

EPISILON_START = 1                          #episilon
EPI_DECAY = 0.995
EPISILON_END = 0.001
##########################################################################
VISUALIZE = False                           #是否观看回放
ENV_MOVE = False                            #env是否变化
VERBOSE = 1                                 #调整日志模式（0\1\2）
##########################################################################

class Cube:
    def __init__(self,size):
        self.size = size
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)
        
    def __str__(self):
        return f'{self.x},{self.y}'
    
    def __sub__(self,other):
        return (self.x-other.x,self.y-other.y)
    
    def __eq__(self,other):
        return self.x == other.x and self.y == other.y
    
    def action(self,choise):
        if choise == 0 :
            self.move(x=1, y=1)
        elif choise == 1 :
            self.move(x=-1, y=1)
        elif choise == 2 :
            self.move(x=1, y=-1)
        elif choise == 3 :
            self.move(x=-1, y=-1)
        elif choise == 4 :
            self.move(x=0, y=1)        
        elif choise == 5 :
            self.move(x=0, y=-1) 
        elif choise == 6 :
            self.move(x=1, y=0) 
        elif choise == 7 :
            self.move(x=-1, y=0)             
        elif choise == 8 :
            self.move(x=0, y=0)             
            
    def move(self,x=False,y=False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
            
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y   
            
        if self.x < 0 :
            self.x = 0
        elif self.x >= self.size :
            self.x = self.size -1

        if self.y < 0 :
            self.y = 0
        elif self.y >= self.size :
            self.y = self.size -1
class envCube:
    SIZE = 10
    OBSERVATION_SPACE_VALUES = (SIZE,SIZE,3)
    ACTION_SPACE_VALUES = 9
    
    FOOD_REWARD = 25
    ENEMY_PENALITY = -300
    MOVE_PENALITY = -1    
    
    d = {1:(255,0,0), #blue
         2:(0,255,0), #green
         3:(0,0,255)} #red

    PLAYER_N = 1
    FOOD_N =2
    ENEMY_N =3    
    
    def reset(self):
        self.player = Cube(self.SIZE)
        self.food = Cube(self.SIZE)
        while self.food == self.player:
            self.food = Cube(self.SIZE)
        
        self.enemy = Cube(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Cube(self.SIZE)
        
        observation = np.array(self.get_image())
        
        self.episode_step = 0
        
        return observation
    
    def step(self,action):
        self.episode_step += 1
        self.player.action(action)
        if ENV_MOVE:
            self.food.move()
            self.enemy.move()

        new_observation = np.array(self.get_image())

        if self.player == self.food :
            reward = self.FOOD_REWARD
        elif self.player == self.enemy :
            reward = self.ENEMY_PENALITY
        else:
            reward = self.MOVE_PENALITY

        done = False
        if self.player == self.food or self.player == self.enemy or self.episode_step>=MAX_STEP:
            done = True
        
        return new_observation,reward,done
    
    def render(self):
        img = self.get_image()       
        img = img.resize((800,800))
        cv2.imshow('Predator',np.array(img))
        cv2.waitKey(1)
    
    def get_image(self):
        env = np.zeros((self.SIZE,self.SIZE,3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        img = Image.fromarray(env,'RGB')
        return img

class ModifiedTensorBoard(TensorBoard):     #调用tensorboard
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
            self.writer.flush()

class DQNAgent():
    def __init__(self):
        self.model = self.create_model()        #构建自己的model
        self.target_model = self.create_model() #构建target_model
        self.target_model.set_weights(self.model.get_weights()) #将自己的model的权重复制给target_model

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)              #构建经验池
        self.update_target_modle_counter = 0        #模型更新次数

        self.tensorboard = ModifiedTensorBoard(log_dir='./logs/dqn_model_{int(time.time())}')   #创建一个自定义的 TensorBoard 对象

    def create_model(self):                                             #构建神经网络模型
        model = Sequential()
        model.add(Conv2D(32,(3,3),activation='relu',input_shape=env.OBSERVATION_SPACE_VALUES))  #添加卷积层
        model.add(Conv2D(32,(3,3),activation='relu'))                   #添加卷积层
        model.add(Flatten())                                            #添加平滑层
        model.add(Dense(32,activation='relu'))                          #添加连接层
        model.add(Dense(env.ACTION_SPACE_VALUES,activation='linear'))   #添加输出层
        model.compile(loss='mse',optimizer='Adam',metrics=['accuracy']) #指定损失函数(均方误差（Mean Squared Error，MmodelSE）)、优化器(动量优化和自适应学习率)和评价函数作为编译器
        return model

    def train(self,terminal_state):
        if len(self.replay_memory) < REPLAY_MEMORY_SIZE:     #经验池容量少时不训练
            return 
        
        minibatch = random.sample(self.replay_memory,MINI_REPLAY_MEMORY_SIZE)   #从经验池中取出若干个transition

        X = []
        y = []

        obs_current = np.array([transition[0] for transition in minibatch])/255 #将一个transition中的obs拿出来，并将像素值缩放到 [0, 1] 范围内
        q_values_current = self.model.predict(obs_current,verbose=0)          #用模型估计当前状态下的 Q 值
        X = obs_current                                             #将当前obs放入X

        obs_new = np.array([transition[3] for transition in minibatch])/255     #将一个transition中的next_obs拿出来，并将像素值缩放到 [0, 1] 范围内
        q_values_future = self.model.predict(obs_new,verbose=0)                #用模型估计将来状态下的 Q 值
        
        for index,(obs,action,reward,new_obs,done) in enumerate(minibatch):
            #X.append(obs)
            if not done:
                yt = reward + DISCOUNT * np.max(np.max(q_values_future[index])) #选取得到q_value最大的action作为动作
            else:
                yt = reward
            
            q_values_current_index = q_values_current[index]
            q_values_current_index[action] = yt
            y.append(q_values_current_index)

        self.model.fit(np.array(X),np.array(y),batch_size=MINI_REPLAY_MEMORY_SIZE,shuffle=False,verbose=0,callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state :
            self.update_target_modle_counter += 1
        
        if self.update_target_modle_counter > UPDATE_TARGET_MODE_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.update_target_modle_counter = 0

    def update_replay_memory(self,transition):
        return self.replay_memory.append(transition)

    def action_q_value_predict(self,obs):               #根据当前状态预测所有action对应的q_value
        return self.model.predict(np.array(obs).reshape(-1,*obs.shape),verbose=0)[0]



env = envCube()                 #创建环境
agent = DQNAgent()              #创建agent

episilon = EPISILON_START        #给episilon赋初值
episode_rewards = []            #创建数组记录每局的reward
model_save_avg_reward = -200    #优秀的模型批判指标

for episode in range(EPISODES):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        if np.random.random() > episilon:
            action = np.argmax(agent.action_q_value_predict(obs))   #选择对应q_value最大的action
        else:
            action = np.random.randint(0,env.ACTION_SPACE_VALUES)

        new_obs,reward,done = env.step(action)

        transition = (obs,action,reward,new_obs,done)               #将参数放入transition
        agent.update_replay_memory(transition)                      #将当前transition放入经验池
        agent.train(done)                                           #训练agent

        obs = new_obs

        episode_reward += reward

        if episode&SHOW_EVERY==0 and VISUALIZE:
            env.render()
    
    episode_rewards.append(episode_reward)
    if episode % STATISTICS_EVERY == 0 and episode >0:
        avg_reward = sum(episode_rewards[-STATISTICS_EVERY:])/len(episode_rewards[-STATISTICS_EVERY:])
        max_reward = max(episode_rewards[-STATISTICS_EVERY:])
        min_reward = min(episode_rewards[-STATISTICS_EVERY:])
        print(f'avg_reward:{avg_reward},max_reward:{max_reward},min_reward:{min_reward}')

        agent.tensorboard.update_stats(avg_reward=avg_reward,max_reward=max_reward,min_reward=min_reward,episilon=episilon,step=episode)

        if min_reward > model_save_avg_reward:          #当每十局的reward超过预设值时，将模型保存下来
            agent.model.save(f'./models/{min_reward:7.3f}_{int(time.time())}.model')
            model_save_avg_reward = avg_reward

    print(f'episode:{episode}      episode_reward:{episode_reward}      episilon:{episilon:7.4f}     ')
    


    episilon *= EPI_DECAY
    episilon = max(episilon,EPISILON_END)

plt.plot([i for i in range(len(episode_rewards))],episode_rewards)
plt.show()
