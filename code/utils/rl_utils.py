# rl.utils
import numpy as np
import torch
import collections
import random
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, terminated): 
        self.buffer.append((state, action, reward, next_state, terminated)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), terminated 

    def size(self): 
        return len(self.buffer)

class Qnet(torch.nn.Module):
    ''' 两层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(Qnet, self).__init__()
        # 构建网络层：输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = torch.nn.Linear(hidden_dims[1], action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层使用ReLU激活函数
        return self.fc3(x)       # 输出层（无激活函数）
    
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        terminateds = torch.tensor(transition_dict['terminateds'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - terminateds)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        # 衰减epsilon,最小值为0.01
        self.epsilon = max(0.001, self.epsilon * 0.999)
        
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

def moving_average(array, window_size):
    ''' 计算移动平均 '''
    cumulative_sum = np.cumsum(np.insert(array, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(array[:window_size-1])[::2] / r
    end = (np.cumsum(array[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                
