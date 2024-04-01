import matplotlib.pyplot as plt
import pandas as pd

def tensorboard_smoothing(x, smooth=0.1):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight += smooth
    return x

fig, ax1 = plt.subplots(1, 1)    # 创建一个带有1x1网格的图形

# 绘制第一条曲线a
len_mean_1 = pd.read_csv("run-DDQN_Algorithum-1-tag-Average_Reward.csv")
ax1.plot(len_mean_1['Step'], tensorboard_smoothing(len_mean_1['Value'], smooth=0.3), color="#3399FF")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Average Episode Length(Episodes)")
ax1.set_title("Average Episode Length")

# 绘制第二条曲线
len_mean_2 = pd.read_csv("run-DDQN_Algorithum-2-tag-Average_Reward.csv")
ax1.plot(len_mean_2['Step'], tensorboard_smoothing(len_mean_2['Value'], smooth=0.3), color="#FF0000")
ax1.set_ylabel("Average Episode Length(Episodes)")

# 绘制第三条曲线
len_mean_3 = pd.read_csv("run-DDQN_Algorithum-3-tag-Average_Reward.csv")
ax1.plot(len_mean_3['Step'], tensorboard_smoothing(len_mean_3['Value'], smooth=0.3), color="#00FF00")
ax1.set_ylabel("Average Episode Length(Episodes)")

# 添加图例说明
ax1.legend(["EKF-DDQN", "PF-DDQN", "DDQN"])

fig.savefig(fname='./figures/ep_len_mean'+'.pdf', format='pdf')
