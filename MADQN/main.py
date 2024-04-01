#可实现三种算法
from DDQN_Algorithum import DDQN 
from Q_learning_Algorithum import Q_learning
from A_Star_Algorithum import A_Star

##########################################################################
EPISODE_N = 60000                          # 总训练局数
REPLAY_MEMORY_SIZE = 5000                   # 经验池的大小
BATCH_SIZE = 100                            # 每次从经验池中取出的个数
DISCOUNT = 0.90                             # 折扣因子
LEARNING_RATE = 0.0001                         # 学习率
UPDATE_TARGET_MODE_EVERY = 10               # model更新频率
STATISTICS_EVERY = 5                        # 记录在tensorboard的频率
MODEL_SAVE_AVG_REWARD = 50                  # 优秀模型评价指标
EPI_START = 1                               # epsilon的初始值
EPI_END = 0.001                             # epsilon的终止值
EPI_DECAY = 0.999995                        # epsilon的保持率
#########################################################################
VERBOSE = 1                                 # 调整日志模式（1——平均游戏得分；2——每局游戏得分）
SHOW_EVERY = 100                            # 显示频率
##########################################################################
demo_1 = DDQN(episodes = EPISODE_N, replay_memory_size = REPLAY_MEMORY_SIZE, batch_size = BATCH_SIZE,discount = DISCOUNT, learning_rate = LEARNING_RATE,update_target_mode_every = UPDATE_TARGET_MODE_EVERY,statistics_every = STATISTICS_EVERY,model_save_avg_reward = MODEL_SAVE_AVG_REWARD,epi_start = EPI_START, epi_end = EPI_END, epi_decay = EPI_DECAY,verbose = VERBOSE,show_every = SHOW_EVERY)
demo_2 = Q_learning(episodes = EPISODE_N, discount = DISCOUNT, learning_rate = LEARNING_RATE,statistics_every = STATISTICS_EVERY,model_save_avg_reward = MODEL_SAVE_AVG_REWARD,epi_start = EPI_START, epi_end = EPI_END, epi_decay = EPI_DECAY,verbose = VERBOSE,show_every = SHOW_EVERY)
demo_3 = A_Star()

#demo_1.train()
demo_2.train()
demo_3.train()

