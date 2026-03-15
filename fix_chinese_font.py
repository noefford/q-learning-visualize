"""
修复 matplotlib 中文显示问题
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 方法1: 使用系统已有的中文字体（推荐）
# Windows 系统通常有 SimHei（黑体）或 Microsoft YaHei（微软雅黑）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 测试中文显示
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 生成测试数据
episodes = np.arange(1000)
rewards = np.random.randn(1000).cumsum() * 0.5 + 5
steps = np.abs(np.random.randn(1000)) * 10 + 5
epsilon = 1.0 * np.exp(-episodes / 200)

# 图1: 奖励曲线
axes[0].plot(episodes, rewards, alpha=0.3, color='blue', label='原始数据')
axes[0].plot(episodes, np.convolve(rewards, np.ones(50)/50, mode='same'), 
             color='red', linewidth=2, label='50轮平均')
axes[0].set_xlabel('轮次 (Episode)')
axes[0].set_ylabel('奖励')
axes[0].set_title('每轮总奖励')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: 步数曲线
axes[1].plot(episodes, steps, alpha=0.3, color='green')
axes[1].plot(episodes, np.convolve(steps, np.ones(50)/50, mode='same'), 
             color='red', linewidth=2)
axes[1].set_xlabel('轮次 (Episode)')
axes[1].set_ylabel('步数')
axes[1].set_title('每轮步数（到达终点或结束）')
axes[1].grid(True, alpha=0.3)

# 图3: 探索率
axes[2].plot(episodes, epsilon, color='purple', linewidth=2)
axes[2].set_xlabel('轮次 (Episode)')
axes[2].set_ylabel('Epsilon (ε)')
axes[2].set_title('探索率衰减')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results_fixed.png', dpi=150, bbox_inches='tight')
print("✓ 已生成修复后的图表: training_results_fixed.png")
plt.show()
