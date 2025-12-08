import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CircuitBatteryEnv(gym.Env):
    """
    电池控制环境建模
    状态空间: [SoC, U1, U2, price, fr_signal】
    动作空间: 11个离散充放电等级
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, price_data=[], fr_data=[], 
                R0=0.0265, R1=0.0352, R2=0.0172, C1=8764.02, C2=1977.84,
                E_rated=0.6, # 储能额定能量容量(MWh)
                Q_rated=800, # 储能额定电量容量(Ah)
                Ns=228, Np=533, # 电池串并联数
                I_min=-1000, I_max=1000, U_min=680, U_max=820, # 储能额定约束
                P_max=0.5, # 最大放电功率(MW)
                OCV_lut=[3.230,3.277,3.311,3.305,3.307,3.313,3.340,3.350,3.351,3.368], 
                SOC_lut=[0.100,0.200,0.300,0.400,0.500,0.600,0.700,0.800,0.900,1.000], 
                soc_min=0.1, soc_max=0.9, 
                delta=50, scene=''):
        super(CircuitBatteryEnv, self).__init__()
        
        # 电路参数
        self.R0 = R0          # 内阻R0
        self.R1 = R1          # 内阻R1
        self.R2 = R2          # 内阻R2
        self.C1 = C1          # 电容C1
        self.C2 = C2          # 电容C2

        # 储能参数
        self.E_rated = E_rated    # 额定能量容量(MWh)
        self.Q_rated = Q_rated    # 额定电量容量(Ah)
        self.Ns = Ns              # 电池串联数
        self.Np = Np              # 电池并联数
        self.P_max = P_max        # 最大放电功率(MW)

        # 约束参数
        self.I_min = I_min    # 最小电流(A)
        self.I_max = I_max    # 最大电流(A)
        self.U_min = U_min    # 最小电压(V)
        self.U_max = U_max    # 最大电压(V)
        self.soc_min = soc_min  # 最小荷电状态
        self.soc_max = soc_max  # 最大荷电状态

        self.OCV_lut = OCV_lut  # 开路电压查找表
        self.SOC_lut = SOC_lut  # 荷电状态查找表

        # 场景设置
        self.scene = scene       # 场景: 'train' 或 'test'
        self._get_timestep()
        self._get_n_steps()

        # 外部数据源
        self.delta = delta      # 调频惩罚系数δ($/MWh)
        self.price_data = np.array(price_data) * 10 # 电价数据预处理($/MWh)
        self.fr_data = np.clip(np.array(fr_data), -1, 1)  # 调频信号数据预处理
        
        # 状态空间定义 (SoC, U1, U2, price, fr_signal)
        U1_min = 1.5 * self.R1 * self.I_min / self.Np
        U1_max = 1.5 * self.R1 * self.I_max / self.Np
        U2_min = 1.5 * self.R2 * self.I_min / self.Np
        U2_max = 1.5 * self.R2 * self.I_max / self.Np
        self.observation_space = spaces.Box(
            low=np.array([self.soc_min, U1_min, U2_min, -np.inf, -1]),
            high=np.array([self.soc_max, U1_max, U2_max, np.inf, 1]),
            dtype=np.float32
        )
        
        # 动作空间定义
        self.action_space = spaces.Discrete(11)  # 11个离散动作
        
        # 初始化状态变量
        self.reset()

    def reset(self, seed=None, options=None, data_head=0):
        super().reset(seed=seed)
        
        # 初始化时间步&数据步
        self.current_step = 0
        self.data_step = data_head
        
        # 初始状态
        self.soc_t = 0.1
        self.U1_t = 0
        self.U2_t = 0
        self.OCV_t = self._get_OCV_single()
        
        return self._get_state(), {}

    def step(self, action):
        # 1. 将离散动作转换为实际功率
        self.I_t = self._action_to_I(action)
        self.I_t_single = self.I_t / self.Np
        self.U_t_single = self._get_U_single()
        self.U_t = self.U_t_single * self.Ns  # 当前电压

        # 2. 计算该步奖励
        reward = self._calculate_reward()

        # 3. 更新状态
        self.soc_t = self.soc_t - self.I_t * (self.timestep / 3600) / self.Q_rated # I为放电电流
        self.OCV_t = self._get_OCV_single()
        self.U1_t = self._get_U1_t_single()
        self.U2_t = self._get_U2_t_single()
        
        # 4. 检查是否结束
        self.current_step += 1
        self.data_step += 1
        done = (self.current_step >= self.n_steps)
        
        # 5. 信息记录
        info = {
            'soc': self.soc_t,
            'voltage': self.U_t,
            'power': self.I_t * self.U_t * 1e-6,
            'energy_cost': self.step_energy_revenue,
            'fr_penalty': self.step_fr_penalty,
        }

        return self._get_state(), reward, done, False, info

    def _action_to_I(self, action):
        # 线性映射：action 0 → I_min, action 10 → I_max
        norm_action = (action - 5) / 5.0  # [-1, 1]
        
        # 考虑SoC约束
        if norm_action >= 0:
            # 放电
            max_I = (self.soc_t - self.soc_min) * self.Q_rated / (self.timestep / 3600)  # 最大放电电流
            I_t = norm_action * self.I_max
            I_t = min(I_t, max_I)
        else:
            # 充电
            min_I = (self.soc_t - self.soc_max) * self.Q_rated / (self.timestep / 3600)  # 最大充电电流
            I_t = -norm_action * self.I_min
            I_t = max(I_t, min_I)
        
        return I_t
    
    def _calculate_reward(self): 
        # 计算当前步的奖励
        self.step_energy_revenue = (self.price_data[self.data_step]) * self.I_t * self.U_t * (1e-6) * (self.timestep / 3600) # 能量成本(I为放电电流)
        P_max = self.U_t * self.I_max * (1e-6)  # 最大放电功率(MW)
        self.step_fr_penalty = self.delta * abs(self.fr_data[self.data_step] * P_max - self.I_t * self.U_t * (1e-6)) * (self.timestep / 3600)  # 调频惩罚
        self.U_outrange_penalty = self._calculate_U_outrange_penalty()  # 电压越界惩罚
        return self.step_energy_revenue - (self.step_fr_penalty + self.U_outrange_penalty)
    
    def _calculate_U_outrange_penalty(self):
        # 电压越界惩罚
        if self.U_t < self.U_min or self.U_t > self.U_max:
            return 1000 * abs(min(self.U_t-self.U_min, self.U_t-self.U_max))  # 惩罚值
        return 0  # 无惩罚
    
    def _get_OCV_single(self):
        # 通过分段线性计算开路电压OCV   
        # 查找SOC所在区间索引idx
        # np.searchsorted返回soc_lut中soc应插入的位置
        idx = np.searchsorted(self.SOC_lut, self.soc_t) - 1
        idx = np.clip(idx, 0, len(self.SOC_lut) - 2)  # 确保不越界
        
        # 获取区间端点
        soc_low, soc_high = self.SOC_lut[idx], self.SOC_lut[idx + 1]
        ocv_low, ocv_high = self.OCV_lut[idx], self.OCV_lut[idx + 1]
        
        # 计算插值权重λ
        if soc_high == soc_low:  # 避免除零
            lamda = 0.0
        else:
            lamda = (self.soc_t - soc_low) / (soc_high - soc_low)
        
        # 线性插值
        ocv = (lamda * ocv_high + (1 - lamda) * ocv_low)
        
        return ocv
    
    def _get_U1_t_single(self):
        # 计算电容C1的电压U1
        U1_t = np.exp(-self.timestep / (self.R1 * self.C1)) * self.U1_t - self.R1 * self.I_t_single * (1 - np.exp(-self.timestep / (self.R1 * self.C1)))

        return U1_t
    
    def _get_U2_t_single(self):
        # 计算电容C2的电压U2
        U2_t = np.exp(-self.timestep / (self.R2 * self.C2)) * self.U2_t - self.R2 * self.I_t_single * (1 - np.exp(-self.timestep / (self.R2 * self.C2)))

        return U2_t
    
    def _get_U_single(self):
        # 计算当前电压U
        return self.OCV_t - self.R0 * self.I_t_single - self.U1_t - self.U2_t
    
    def _get_state(self):
        # 获取当前状态
        idx=min(self.data_step, len(self.price_data)-1)
        return np.array([
            self.soc_t,              # c_t
            self.U1_t,              # U1_t
            self.U2_t,              # U2_t
            self.price_data[idx],  # p_t
            self.fr_data[idx],     # f_t
        ], dtype=np.float32)
    
    def _get_timestep(self):
        # 根据模式设定时间步长
        if self.scene == 'train':
            self.timestep = 10
        elif self.scene == 'test':
            self.timestep = 2
            
    def _get_n_steps(self):
        # 根据模式设定episode步数
        if self.scene == 'train':
            self.n_steps = 8640
        elif self.scene == 'test':
            self.n_steps = 43200
    
    def render(self):
        # 调试用,显示各项数值
        print(f"Step: {self.current_step-1}, I: {self.I_t:.4f}, SoC: {self.soc_t:.5f}, OCV:{self.OCV_t:.4f}, U1: {self.U1_t:.4f}, U2: {self.U2_t:.4f}, U: {self.U_t:.4f}, "
              f"Price: {self.price_data[self.data_step-1]:.2f}, FR: {self.fr_data[self.data_step-1]:.4f}, "
              f"Energy Cost: {self.step_energy_revenue}, "
              f"FR Penalty: {self.step_fr_penalty}, ")

    def close(self):
        pass