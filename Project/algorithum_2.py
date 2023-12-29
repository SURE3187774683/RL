#Q-learning算法

import numpy as np
import time
import os
import pickle
from matplotlib import style
style.use('ggplot')

import torch
from tensorboardX import SummaryWriter
import numpy as np
from env import envCube

class Algorithum_2:
    def __init__(self, episodes, replay_memory_size, batch_size, discount, learning_rate, update_target_mode_every,
                 statistics_every, model_save_avg_reward, epi_start, epi_end, epi_decay, visualize, verbose, show_every):

        path = os.path.realpath(__file__)
        filename = os.path.splitext(os.path.basename(path))[0]
        self.writer = SummaryWriter(f'logs/{filename}')
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
        env = envCube()
        q_table = env.get_qtable()
        q_table = {k: torch.tensor(v).to(self.device) for k, v in q_table.items()}
        episode_rewards = []
        epsilon = self.epi_start

        for episode in range(self.episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if np.random.random() > epsilon:
                    action = np.argmax(q_table[obs].cpu().numpy())
                else:
                    action = np.random.randint(0, env.ACTION_SPACE_VALUES)

                new_obs, reward, done = env.step(action)

                # Update the Q_table
                current_q = q_table[obs][action]
                max_future_q = torch.max(q_table[new_obs])

                if reward == env.FOOD_REWARD:
                    new_q = torch.tensor(env.FOOD_REWARD, dtype=torch.float32).to(self.device)
                else:
                    new_q = current_q + self.learning_rate * (reward + self.discount * max_future_q - current_q)
                    new_q = new_q.to(self.device)
                epsilon = max(self.epi_end, epsilon * self.epi_decay)

                q_table[obs][action] = new_q.to(self.device)  # 将更新后的值移动到GPU上
                obs = new_obs
                episode_reward += reward
                
                if done:
                    episode_rewards.append(episode_reward)
                    break

            if episode % self.show_every == 0:
                print(f"Episode: {episode}        Epsilon:{epsilon}")
                if self.verbose == 1:
                    avg_reward = np.mean(episode_rewards)
                    print(f"### Average Reward: {avg_reward}")
                if self.verbose == 2:
                    print(f"### Episode Reward: {episode_rewards[-1]}")
                if self.visualize:
                    env.render()

            if episode % self.statistics_every == 0:
                avg_reward = sum(episode_rewards[-self.statistics_every:]) / self.statistics_every
                max_reward = max(episode_rewards[-self.statistics_every:])
                min_reward = min(episode_rewards[-self.statistics_every:])
                self.writer.add_scalar('Episode Reward', episode_reward, episode)
                self.writer.add_scalar('Average Reward', avg_reward, episode)
                self.writer.add_scalar('Max Reward', max_reward, episode)
                self.writer.add_scalar('Min Reward', min_reward, episode)
                self.writer.add_scalar('Epsilon', epsilon, episode)

                if avg_reward > self.model_save_avg_reward and avg_reward < 90:          #保存优秀的模型

                    env.render_trajectory(2)  # 保存智能体轨迹图像
                    self.model_save_avg_reward = avg_reward

#with open(f'qtable_{int(time.time())}.pickle','wb') as f:
#    pickle.dump(q_table,f)
