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
    def __init__(self,episodes, replay_memory_size, batch_size,discount, learning_rate,update_target_mode_every,statistics_every,model_save_avg_reward,epi_start, epi_end, epi_decay,visualize,verbose,show_every):
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
        self.visualize = visualize
        self.verbose = verbose
        self.show_every = show_every

    def train(self):
        player_positions = [(0,0),(25,0)]
        
        env = envCube(player_positions)

        agents = []
        for player_id in range(len(player_positions)):
            agent = DQNAgent(env.OBSERVATION_SPACE_VALUES, env.ACTION_SPACE_VALUES, self.   replay_memory_size, self.batch_size, self.discount,self.learning_rate , self.  epi_start, self.epi_end, self.epi_decay,self.device,player_id)
            agents.append(agent)

        for episode in range(self.episodes): 
            state = env.reset()                                 #重置环境
            done = False
            episode_rewards = [0] * env.NUM_PLAYERS                                  #每局奖励清零

            while not done:
                for agent in agents:
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    agent.push_transition(state, action, reward, next_state, done)
                    agent.update_epsilon()
                    agent.update_model()
                    state = next_state
                    episode_rewards[agent.player_id] += reward

                    if done:
                        agent.episode_rewards.append(episode_rewards[agent.player_id])
                        break
            env.render_trajectory(1)
                
            if episode % self.update_target_mode_every == 0:         #更新target_model(将当前模型的复制到目标模型)
                for agent in agents:
                    agent.update_target_model()
                
            if episode%self.show_every==0:                           #打印日志
                print(f"Episode: {episode}        Epsilon:{agent.epsilon}")
                if self.verbose == 1:                                #输出平均奖励
                    print(f"### Average Reward: {np.mean(agent.episode_rewards)}")                
                if self.verbose == 2:                                #输出每轮游戏的奖励
                    print(f"### Episode Reward: {agent.episode_rewards[-1]}")
                if self.visualize:                                   #显示动画
                    env.render()

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