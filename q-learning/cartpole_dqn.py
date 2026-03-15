"""
DQN (Deep Q-Network) 实现 - CartPole 环境
使用神经网络近似Q函数，解决连续状态空间问题
"""
import numpy as np
import random
from collections import deque
from typing import List, Tuple

# 尝试导入tensorflow，如果没有则提示安装
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("警告: 未安装 TensorFlow，请先安装: pip install tensorflow")
    print("这个文件展示了DQN的实现，但需要TensorFlow才能运行")
    # 创建占位符类，避免导入错误
    class MockLayer:
        def Dense(self, *args, **kwargs):
            pass
    class MockKeras:
        layers = MockLayer()
        def Sequential(self):
            pass
        def Input(self, *args, **kwargs):
            pass
    keras = MockKeras()


class ReplayBuffer:
    """
    经验回放缓冲区 (Experience Replay)
    
    作用:
    - 存储智能体的经验 (state, action, reward, next_state, done)
    - 随机采样打破数据相关性，提高训练稳定性
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加一条经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """随机采样一批经验"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN 智能体
    
    核心改进:
    1. 使用神经网络近似Q函数（处理连续状态空间）
    2. 经验回放缓冲区
    3. 目标网络（Target Network）稳定训练
    
    算法流程:
    1. 初始化Q网络和目标Q网络
    2. 对于每个时间步:
       - 选择动作（ε-贪心）
       - 执行动作，存储经验
       - 从缓冲区采样，计算TD误差
       - 更新Q网络
       - 定期同步目标网络
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 10
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(capacity=buffer_size)
        
        # 构建Q网络和目标网络
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # 训练步数计数
        self.train_step = 0
    
    def _build_model(self):
        """
        构建神经网络模型
        
        网络结构:
        - 输入层: state_size (状态维度)
        - 隐藏层: 64个神经元，ReLU激活
        - 隐藏层: 64个神经元，ReLU激活
        - 输出层: action_size (每个动作的Q值)
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')  # 线性输出Q值
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss='mse'  # 均方误差
        )
        
        return model
    
    def update_target_model(self):
        """
        将主网络的权重复制到目标网络
        目标网络用于计算TD目标，提高稳定性
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-贪心策略）
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # 使用神经网络预测Q值，选择最大值
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """
        从经验回放缓冲区采样并训练网络
        """
        # 缓冲区中数据不足时不训练
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        minibatch = self.memory.sample(self.batch_size)
        
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])
        
        # 当前Q值
        current_q = self.model.predict(states, verbose=0)
        
        # 下一状态的Q值（使用目标网络计算）
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # 计算目标Q值
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                # TD目标: r + γ * max(Q'(s'))
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # 训练网络
        self.model.fit(states, target_q, epochs=1, verbose=0)
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.train_step += 1
        
        # 定期更新目标网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()
    
    def save(self, filepath: str):
        """保存模型"""
        self.model.save(filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        self.model = keras.models.load_model(filepath)
        self.update_target_model()
        print(f"模型已从 {filepath} 加载")


def train_dqn_cartpole(n_episodes: int = 500):
    """
    训练DQN解决CartPole问题
    
    CartPole环境简介:
    - 一个杆子连接在小车上，小车可以左右移动
    - 目标: 保持杆子不倒（角度不超过一定范围）
    - 状态: 小车位置、速度、杆子角度、角速度（4维连续值）
    - 动作: 左移(0)或右移(1)
    - 奖励: 每存活一帧获得+1
    """
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError:
            print("请先安装 gymnasium: pip install gymnasium")
            return
    
    # 创建环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n             # 2
    
    print("=" * 60)
    print("DQN 训练 CartPole-v1")
    print("=" * 60)
    print(f"状态空间: {state_size} 维 (位置, 速度, 角度, 角速度)")
    print(f"动作空间: {action_size} (左移, 右移)")
    print("=" * 60)
    
    # 创建智能体
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=10
    )
    
    scores = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()  # gymnasium 返回 (state, info)
        score = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.choose_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 修改奖励，鼓励长时间存活
            reward = reward if not done else -10
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            score += 1
        
        # 训练（经验回放）
        agent.replay()
        
        scores.append(score)
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"轮次: {episode + 1}/{n_episodes}, 平均分数: {avg_score:.1f}, "
                  f"探索率: {agent.epsilon:.3f}, 缓冲区大小: {len(agent.memory)}")
    
    env.close()
    
    # 保存模型
    agent.save("dqn_cartpole.h5")
    
    print("\n训练完成！")
    print(f"最后50轮的平均分数: {np.mean(scores[-50:]):.1f}")
    
    return agent, scores


if __name__ == "__main__":
    # 训练DQN
    train_dqn_cartpole(n_episodes=500)
