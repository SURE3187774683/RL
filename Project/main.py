#训练两种算法
##########################################################################
EPISODE_N = 1000                           #总训练局数
REPLAY_MEMORY_SIZE = 64                   #经验池的大小
BATCH_SIZE = 20                            #每次从经验池中取出的个数
DISCOUNT = 0.95                             #折扣因子
LEARNING_RATE = 1e-3                        #学习率(步长)
UPDATE_TARGET_MODE_EVERY = 20               #model更新频率
STATISTICS_EVERY = 20                       #记录在tensorboard的频率
MODEL_SAVE_AVG_REWARD = 130                 #优秀模型评价指标
EPI_START = 1                               #epsilon的初始值
EPI_END = 0.001                             #epsilon的终止值
EPI_DECAY = 0.9995                         #epsilon的缩减速率
#########################################################################
VISUALIZE = False                           #是否观看回放
VERBOSE = 1                                 #调整日志模式（1——平均游戏得分；2——每局游戏得分）
SHOW_EVERY = 50                            #显示频率
##########################################################################

from algorithum_1 import Algorithum_1 
from algorithum_2 import Algorithum_2

demo_1 = Algorithum_1(episodes = EPISODE_N, replay_memory_size = REPLAY_MEMORY_SIZE, batch_size = BATCH_SIZE,discount = DISCOUNT, learning_rate = LEARNING_RATE,update_target_mode_every = UPDATE_TARGET_MODE_EVERY,statistics_every = STATISTICS_EVERY,model_save_avg_reward = MODEL_SAVE_AVG_REWARD,epi_start = EPI_START, epi_end = EPI_END, epi_decay = EPI_DECAY,visualize = VISUALIZE,verbose = VERBOSE,show_every = SHOW_EVERY)

demo_2 = Algorithum_2(episodes = EPISODE_N, replay_memory_size = REPLAY_MEMORY_SIZE, batch_size = BATCH_SIZE,discount = DISCOUNT, learning_rate = LEARNING_RATE,update_target_mode_every = UPDATE_TARGET_MODE_EVERY,statistics_every = STATISTICS_EVERY,model_save_avg_reward = MODEL_SAVE_AVG_REWARD,epi_start = EPI_START, epi_end = EPI_END, epi_decay = EPI_DECAY,visualize = VISUALIZE,verbose = VERBOSE,show_every = SHOW_EVERY)

demo_1.train()
demo_2.train()
