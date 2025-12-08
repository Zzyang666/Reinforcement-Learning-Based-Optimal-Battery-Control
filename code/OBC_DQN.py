import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import warnings
from utils.rl_utils import ReplayBuffer, DQN, moving_average
from OBC_linear_env import LinearBatteryEnv
from OBC_circuit_env import CircuitBatteryEnv
from utils.data_loader import prepare_data
# 忽略特定警告
warnings.filterwarnings('ignore', message='Creating a tensor from a list of numpy.ndarrays')

# 添加OpenMP环境变量设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 超参数设置
lr = 1e-4
num_episodes = 1000
episode_steps = 8640    
hidden_dims = [128, 32]
gamma = 0.95
epsilon = 1e-2
target_update = 8640
buffer_size = 8640*14
minimal_size = 8640
batch_size = 256
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 数据导入&创建环境
fr_files="data/fr_data/04 2024.xlsx"
price_files="data/price_data/04 2024.xlsx"
(price_data, fr_data) = prepare_data(fr_files, price_files, mode="train", point="HB_NORTH", 
                                     fr_start_date="2024/4/7", price_start_date="2024-04-07")
env_name = 'Battery-Trading'
env = CircuitBatteryEnv(price_data=price_data, fr_data=fr_data, scene='train')

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
env.reset(seed=42)
env.action_space.seed(42)

replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dims, action_dim, lr, gamma, epsilon, target_update, device)


# 主训练循环
return_list = []
for i_episode in range(num_episodes):
    episode_return = 0
    state, _ = env.reset(data_head=(i_episode % 7) * episode_steps)
    done = False
    
    with tqdm(total=env.n_steps, desc=f'Episode {i_episode+1}/{num_episodes}', ncols=100) as pbar:
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            env.render()
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, terminated)
            state = next_state
            episode_return += reward

            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'terminateds': b_d
                }
                agent.update(transition_dict)

            pbar.set_postfix({'total_reward': f'{episode_return:.2f}',})
            pbar.update(1)

    return_list.append(episode_return)

env.close()

# 绘制训练曲线
if len(return_list) > 0:
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.savefig('training_results/returns.png')
    plt.show()

    # 计算并打印平均回报
    mv_return = moving_average(return_list, 7)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.savefig('training_results/average_returns.png')
    plt.show()
