"""
Q-Learning 智能体实现
Q-Learning 是一种经典的无模型强化学习算法
"""
import numpy as np
import pickle
from typing import List


class QLearningAgent:
    """
    Q-Learning 智能体
    
    核心思想:
    - Q表: 存储每个状态下每个动作的价值估计
    - 选择动作: ε-贪心策略（探索 vs 利用）
    - 更新Q值: 使用TD(0)误差更新
    
    Q值更新公式:
    Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    
    其中:
    - α (alpha): 学习率
    - γ (gamma): 折扣因子（未来奖励的重要性）
    - r: 即时奖励
    - s': 下一个状态
    - max(Q(s',a')): 下一状态的最大Q值（贝尔曼最优方程）
    """
    
    def __init__(
        self,
        n_states: int,       # 状态空间大小
        n_actions: int,      # 动作空间大小
        learning_rate: float = 0.1,    # 学习率 α
        gamma: float = 0.99,           # 折扣因子 γ
        epsilon: float = 1.0,          # 初始探索率 ε
        epsilon_decay: float = 0.995,  # 探索率衰减
        epsilon_min: float = 0.01      # 最小探索率
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表：每个状态-动作对的初始Q值为0
        self.q_table = np.zeros((n_states, n_actions))
        
        # 记录训练信息
        self.training_step = 0
        
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        选择动作（ε-贪心策略）
        
        Args:
            state: 当前状态
            training: 是否在训练模式
        
        Returns:
            选择的动作
        
        策略说明:
        - 以 ε 的概率随机选择动作（探索）
        - 以 1-ε 的概率选择当前Q值最高的动作（利用）
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择Q值最大的动作
            # 如果有多个相同最大值，随机选择其中一个
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            # 找到所有具有最大Q值的动作
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        更新Q值（核心学习算法）
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # 当前Q值
        current_q = self.q_table[state, action]
        
        # 计算TD目标值
        if done:
            # 如果结束，没有下一个状态，TD目标就是即时奖励
            td_target = reward
        else:
            # 否则，TD目标 = 即时奖励 + 折扣 * 下一状态的最大Q值
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # TD误差 = TD目标 - 当前Q值
        td_error = td_target - current_q
        
        # 更新Q值: Q = Q + α * TD误差
        self.q_table[state, action] += self.lr * td_error
        
        self.training_step += 1
        
    def update_epsilon(self):
        """
        衰减探索率
        随着训练进行，逐渐减少随机探索，增加利用
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def save(self, filepath: str):
        """保存Q表到文件"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """从文件加载Q表"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.training_step = data['training_step']
        print(f"模型已从 {filepath} 加载")
    
    def get_q_table(self) -> np.ndarray:
        """获取Q表（用于查看学习结果）"""
        return self.q_table
    
    def get_best_action(self, state: int) -> int:
        """获取某个状态下的最佳动作（贪婪策略）"""
        return np.argmax(self.q_table[state])


# 简单的测试
if __name__ == "__main__":
    agent = QLearningAgent(n_states=16, n_actions=4)
    
    # 模拟一个学习步骤
    state = 0
    action = agent.choose_action(state)
    print(f"状态 {state} 选择动作: {action}")
    
    # 模拟更新
    agent.learn(state=0, action=action, reward=1.0, next_state=1, done=False)
    print(f"Q表更新后的部分值:\n{agent.q_table[:2]}")
