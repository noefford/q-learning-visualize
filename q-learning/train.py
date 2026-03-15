"""
训练脚本：训练 Q-Learning 智能体解决网格世界问题
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from grid_world_env import GridWorldEnv
from q_learning_agent import QLearningAgent

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def train(
    n_episodes: int = 1000,     # 训练轮数
    max_steps: int = 100,        # 每轮最大步数
    print_interval: int = 100,   # 打印间隔
    save_model: bool = True      # 是否保存模型
):
    """
    训练 Q-Learning 智能体
    
    训练过程:
    1. 初始化环境和智能体
    2. 对于每一轮（episode）:
       - 重置环境
       - 循环执行直到结束或达到最大步数:
         * 根据当前状态选择动作
         * 执行动作，观察新状态和奖励
         * 更新Q值
       - 衰减探索率
       - 记录本轮奖励
    3. 保存模型并可视化训练结果
    """
    
    # 创建环境
    env = GridWorldEnv(size=4)
    
    # 创建智能体
    agent = QLearningAgent(
        n_states=env.observation_space,
        n_actions=env.action_space,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,        # 初始完全随机探索
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("=" * 60)
    print("开始训练 Q-Learning 智能体")
    print("=" * 60)
    print(f"环境设置: {env.size}x{env.size} 网格世界")
    print(f"训练轮数: {n_episodes}")
    print(f"学习率: {agent.lr}, 折扣因子: {agent.gamma}")
    print(f"初始探索率: {agent.epsilon}, 最小探索率: {agent.epsilon_min}")
    print("=" * 60)
    
    # 记录训练数据
    episode_rewards = []    # 每轮的总奖励
    episode_steps = []      # 每轮的步数
    epsilon_history = []    # 探索率历史
    
    for episode in range(n_episodes):
        state = env.reset()  # 重置环境
        total_reward = 0
        steps = 0
        done = False
        
        # 执行一轮
        while not done and steps < max_steps:
            # 选择动作
            action = agent.choose_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 学习（更新Q值）
            agent.learn(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            total_reward += reward
            steps += 1
        
        # 衰减探索率
        agent.update_epsilon()
        
        # 记录数据
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        epsilon_history.append(agent.epsilon)
        
        # 定期打印进度
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_steps = np.mean(episode_steps[-print_interval:])
            print(f"轮次: {episode + 1}/{n_episodes}")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均步数: {avg_steps:.2f}")
            print(f"  探索率 ε: {agent.epsilon:.4f}")
            print("-" * 40)
    
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 保存模型
    if save_model:
        agent.save("q_learning_model.pkl")
    
    # 绘制训练曲线
    plot_training_results(episode_rewards, episode_steps, epsilon_history, n_episodes)
    
    return agent, env, episode_rewards


def plot_training_results(rewards, steps, epsilon_history, n_episodes):
    """
    绘制训练过程的可视化图表
    """
    # 平滑曲线（移动平均）
    window = 50
    if len(rewards) >= window:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smoothed_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
    else:
        smoothed_rewards = rewards
        smoothed_steps = steps
    
    plt.figure(figsize=(15, 5))
    
    # 子图1: 每轮奖励
    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.3, label='原始数据', color='blue')
    if len(rewards) >= window:
        plt.plot(range(window-1, len(rewards)), smoothed_rewards, label=f'{window}轮平均', color='red')
    plt.xlabel('轮次 (Episode)')
    plt.ylabel('总奖励')
    plt.title('每轮总奖励')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 每轮步数
    plt.subplot(1, 3, 2)
    plt.plot(steps, alpha=0.3, color='green')
    if len(steps) >= window:
        plt.plot(range(window-1, len(steps)), smoothed_steps, color='red', linewidth=2)
    plt.xlabel('轮次 (Episode)')
    plt.ylabel('步数')
    plt.title('每轮步数（到达终点或结束）')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 探索率变化
    plt.subplot(1, 3, 3)
    plt.plot(epsilon_history, color='purple')
    plt.xlabel('轮次 (Episode)')
    plt.ylabel('Epsilon (ε)')
    plt.title('探索率衰减')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    print("训练结果图已保存为 'training_results.png'")


if __name__ == "__main__":
    # 可以调整参数进行训练
    agent, env, rewards = train(
        n_episodes=1000,      # 训练1000轮
        max_steps=100,        # 每轮最多100步
        print_interval=100,   # 每100轮打印一次
        save_model=True
    )
