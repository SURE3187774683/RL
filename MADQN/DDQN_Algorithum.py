#DDQN算法

import torch
import os
import time
from tensorboardX import SummaryWriter
import numpy as np
from agent import DQNAgent
from agent import DQN
from env import envCube

class DDQN:
    def __init__(self,episodes, replay_memory_size, batch_size,discount, learning_rate,update_target_mode_every,statistics_every,model_save_avg_reward,epi_start, epi_end, epi_decay,verbose,show_every):
        self.path = os.path.realpath(__file__)
        self.filename = os.path.splitext(os.path.basename(self.path))[0]
        self.writer = SummaryWriter(f'logs/{self.filename}')                  #创建笔
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.episodes = episodes
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.discount = discount
        self.learning_rate = learning_rate
        self.update_target_mode_every = update_target_mode_every
        self.statistics_every = statistics_every
        self.model_save_avg_reward = model_save_avg_reward
        self.epi_start = epi_start
        self.epi_end = epi_end
        self.epi_decay = epi_decay
        self.verbose = verbose
        self.show_every = show_every

        self.agents = []
        self.player_positions = [(0,0)]
        self.enemy_positions  = [(5,8),(6,8), 
                                 (5,9),(6,9),
                                 (5,10),(6,10),
         
                                 (9,2), (10,2),(11,2),(12,2),(13,2), (14,2),(15,2),(16,2),(17,2),(9,3),(9,4),
                                 
                                 (22,5), (23,5),(24,5),
                                 (22,6), (23,6),(24,6),
                                 (22,7), (23,7),(24,7), 
                                 (22,8), (23,8),(24,8),
                                 (22,9), (23,9),(24,9),
         
                                 (11,10),(12,10),(13,10),(14,10),(15,10),(16,10),
                                 (11,11),(12,11),
                                 (11,12),(12,12),
                                 (11,13),(12,13),
                                 
                                 (3,20),(4,20),(5,20),
                                 (3,21),(4,21),(5,21),(3,23),(3,24),(3,25),
                                 (3,22),(4,22),(5,22),
         
                                 (14,20), (14,21),(14,22),(14,23), (14,24),(14,25),(14,26),(14,27),
                                                  (15,22),(15,23), (15,24),(15,25),(15,26),(15,27),
                                 
                                 (21,17), (22,17),(23,17),(24,17), (25,17),(26,17),
                                 (21,18), (22,18),(23,18),(24,18), (25,18),(26,18),(26,19),
                                 (26,20),(26,21),(26,22),(26,23),(26,24)]  # 指定enemy的位置
    def train(self):
        env = envCube(self.player_positions,self.enemy_positions)

        for player_id in range(len(self.player_positions)):
            agent = DQNAgent(player_id, env.OBSERVATION_SPACE_VALUES, env.ACTION_SPACE_VALUES, self.replay_memory_size, self.batch_size, self.discount,self.learning_rate , self.epi_start, self.epi_end, self.epi_decay,self.device)
            self.agents.append(agent)

        for episode in range(self.episodes): 
            state = env.reset()                                 #重置环境
            done = False
            episode_rewards = [0] * env.NUM_PLAYERS                                  #每局奖励清零
            while not done:
                for agent in self.agents:
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(agent.player_id,action)
                    agent.push_transition(state, action, reward, next_state, done)
                    agent.update_epsilon()
                    agent.update_model()
                    state = next_state
                    episode_rewards[agent.player_id] += reward

                    if done:
                        agent.episode_rewards.append(episode_rewards[agent.player_id])
                        break
            
                
            if episode % self.update_target_mode_every == 0:         #更新target_model(将当前模型的复制到目标模型)
                for agent in self.agents:
                    agent.update_target_model()
                
            if episode%self.show_every==0:                           #打印日志
                env.render_trajectory(1)
                print(f"Episode: {episode}        Epsilon:{agent.epsilon}")
                if self.verbose == 1:                                #输出平均奖励
                    print(f"### Average Reward: {np.mean(agent.episode_rewards)}")                
                if self.verbose == 2:                                #输出每轮游戏的奖励
                    print(f"### Episode Reward: {agent.episode_rewards[-1]}")

            if episode % self.statistics_every == 0:                 #记录有用的参数
                avg_reward = sum(agent.episode_rewards[-self.statistics_every:])/len(agent.episode_rewards[-self.statistics_every:])
                max_reward = max(agent.episode_rewards[-self.statistics_every:])
                min_reward = min(agent.episode_rewards[-self.statistics_every:])

                total_episode_reward = sum(episode_rewards)
                self.writer.add_scalar('Episode Reward', total_episode_reward, episode)

                self.writer.add_scalar('Average Reward', avg_reward, episode)
                self.writer.add_scalar('Max Reward', max_reward, episode)
                self.writer.add_scalar('Min Reward', min_reward, episode)
                self.writer.add_scalar('Epsilon', agent.epsilon, episode)
                self.writer.add_scalar('Loss', agent.loss_value, episode)

                if avg_reward > self.model_save_avg_reward and avg_reward < 90:          #保存优秀的模型
                    self.model_save_avg_reward = avg_reward
                    model_dir = f'./models/{self.filename}'
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    model_path = os.path.join(model_dir, f'{avg_reward:7.3f}_{int(time.time())}.model')
                    torch.save(DQN(env.OBSERVATION_SPACE_VALUES,env.ACTION_SPACE_VALUES).state_dict(), model_path)
