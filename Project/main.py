#Double-DQN实现pathplanning
#SURE   

import torch
import os
import time
from tensorboardX import SummaryWriter
import numpy as np
from agent import DQNAgent
from agent import DQN
from env import envCube

##########################################################################
EPISODE_N = 100000                           #总训练局数
REPLAY_MEMORY_SIZE = 1000                    #经验池的大小
BATCH_SIZE = 500                             #每次从经验池中取出的个数
DISCOUNT = 0.95                             #折扣因子
LEARNING_RATE = 1e-3                        #学习率(步长)
UPDATE_TARGET_MODE_EVERY = 20               #model更新频率
STATISTICS_EVERY = 20                       #记录在tensorboard的频率
MODEL_SAVE_AVG_REWARD = 130                 #优秀模型评价指标
EPI_START = 1                               #epsilon的初始值
EPI_END = 0.001                             #epsilon的终止值
EPI_DECAY = 0.99995                           #epsilon的缩减速率
#########################################################################
VISUALIZE = False                           #是否观看回放
VERBOSE = 1                                 #调整日志模式（1——平均游戏得分；2——每局游戏得分）
SHOW_EVERY = 100                            #显示频率
##########################################################################

env = envCube()
agent = DQNAgent(env.OBSERVATION_SPACE_VALUES, env.ACTION_SPACE_VALUES, REPLAY_MEMORY_SIZE, BATCH_SIZE, DISCOUNT,LEARNING_RATE , EPI_START, EPI_END, EPI_DECAY)
writer = SummaryWriter('logs/')                         #创建笔
for episode in range(EPISODE_N): 
    state = env.reset()                                 #重置环境
    done = False
    episode_reward = 0                                  #每局奖励清零

    while not done:
        action = agent.select_action(state)             #选择action
        next_state, reward, done = env.step(action)     #游戏走一步
        agent.push_transition(state, action, reward, next_state, done)   #将当前状态放入池       
        agent.update_epsilon()                          #更新epsilon
        state = next_state                              #更新state
        episode_reward += reward                        #累加当次训练的reward

        if done:
            agent.episode_rewards.append(episode_reward)#收集所有训练累计的rewar
            break
        agent.update_model()                            #更新model

    if episode % UPDATE_TARGET_MODE_EVERY == 0:         #更新target_model(将当前模型的复制到目标模型)
        agent.update_target_model()
    if episode%SHOW_EVERY==0:                           #打印日志
        print(f"Episode: {episode}        Epsilon:{agent.epsilon}")
        if VERBOSE == 1:                                #输出平均奖励
            print(f"### Average Reward: {np.mean(agent.episode_rewards)}")                
        if VERBOSE == 2:                                #输出每轮游戏的奖励
            print(f"### Episode Reward: {agent.episode_rewards[-1]}")
        if VISUALIZE:                                   #显示动画
            env.render()

    if episode % STATISTICS_EVERY == 0:                 #记录有用的参数
        avg_reward = sum(agent.episode_rewards[-STATISTICS_EVERY:])/len(agent.episode_rewards[-STATISTICS_EVERY:])
        max_reward = max(agent.episode_rewards[-STATISTICS_EVERY:])
        min_reward = min(agent.episode_rewards[-STATISTICS_EVERY:])
        writer.add_scalar('Episode Reward', episode_reward, episode)
        writer.add_scalar('Average Reward', avg_reward, episode)
        writer.add_scalar('Max Reward', max_reward, episode)
        writer.add_scalar('Min Reward', min_reward, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        writer.add_scalar('Loss', agent.loss_value, episode)
        
        if avg_reward > MODEL_SAVE_AVG_REWARD:          #保存优秀的模型
            MODEL_SAVE_AVG_REWARD = avg_reward
            model_dir = './models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, f'{avg_reward:7.3f}_{int(time.time())}.model')
            torch.save(DQN(env.OBSERVATION_SPACE_VALUES,env.ACTION_SPACE_VALUES).state_dict(), model_path)
