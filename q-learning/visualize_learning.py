"""
可视化学习过程：展示Q表如何随着训练逐步收敛
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from grid_world_env import GridWorldEnv
from q_learning_agent import QLearningAgent

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def visualize_q_values(q_table, env, title="Q值可视化"):
    """
    使用热力图展示每个状态-动作对的Q值
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 创建4个子图，分别展示4个动作的Q值
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=16)
    
    actions = ["上 (Up)", "下 (Down)", "左 (Left)", "右 (Right)"]
    
    for idx, (ax, action_name) in enumerate(zip(axes.flat, actions)):
        # 重塑Q值为网格形状
        q_values = q_table[:, idx].reshape(env.size, env.size)
        
        # 绘制热力图
        im = ax.imshow(q_values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        # 设置标题
        ax.set_title(action_name, fontsize=14)
        
        # 设置刻度
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.set_xticklabels(range(env.size))
        ax.set_yticklabels(range(env.size))
        
        # 在每个格子中显示数值
        for i in range(env.size):
            for j in range(env.size):
                text = ax.text(j, i, f'{q_values[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=10)
        
        # 标记特殊位置
        ax.plot(env.goal_pos[1], env.goal_pos[0], 'r*', markersize=20, label='Goal')
        for trap in env.traps:
            ax.plot(trap[1], trap[0], 'rx', markersize=15, label='Trap')
        
        ax.set_xlabel('列 (Column)')
        ax.set_ylabel('行 (Row)')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1)
    cbar.set_label('Q值', fontsize=12)
    
    plt.tight_layout()
    return fig


def visualize_policy_grid(q_table, env, title="策略可视化"):
    """
    在网格上可视化最优策略（箭头表示最佳动作）
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 使(0,0)在左上角
    
    # 动作对应的箭头
    arrows = {
        0: (0, -0.3),    # 上
        1: (0, 0.3),     # 下
        2: (-0.3, 0),    # 左
        3: (0.3, 0)      # 右
    }
    
    # 绘制网格
    for i in range(env.size):
        for j in range(env.size):
            state = i * env.size + j
            pos = (i, j)
            
            # 绘制格子
            if pos == env.goal_pos:
                color = '#90EE90'  # 浅绿色（目标）
                label = 'G'
            elif pos in env.traps:
                color = '#FFB6C1'  # 浅红色（陷阱）
                label = 'X'
            elif pos == env.start_pos:
                color = '#87CEEB'  # 浅蓝色（起点）
                label = 'S'
            else:
                color = 'white'
                label = ''
            
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # 在格子中显示标签
            if label:
                ax.text(j, i, label, ha='center', va='center', 
                       fontsize=20, fontweight='bold')
            
            # 如果不是终点或陷阱，绘制最佳动作箭头
            if pos not in env.traps and pos != env.goal_pos:
                best_action = np.argmax(q_table[state])
                dx, dy = arrows[best_action]
                ax.arrow(j, i, dx, dy, head_width=0.15, head_length=0.1, 
                        fc='black', ec='black', linewidth=2)
    
    # 设置刻度和标签
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xticklabels(range(env.size))
    ax.set_yticklabels(range(env.size))
    ax.set_xlabel('列 (Column)', fontsize=12)
    ax.set_ylabel('行 (Row)', fontsize=12)
    ax.set_title(title, fontsize=16)
    
    # 添加图例
    goal_patch = mpatches.Patch(color='#90EE90', label='目标 (Goal)')
    trap_patch = mpatches.Patch(color='#FFB6C1', label='陷阱 (Trap)')
    start_patch = mpatches.Patch(color='#87CEEB', label='起点 (Start)')
    ax.legend(handles=[goal_patch, trap_patch, start_patch], 
             loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def demonstrate_learning_process():
    """
    演示Q-learning的学习过程，展示不同训练阶段的Q表变化
    """
    # 创建环境和智能体
    env = GridWorldEnv(size=4)
    agent = QLearningAgent(
        n_states=env.observation_space,
        n_actions=env.action_space,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # 记录不同阶段的Q表
    stages = [0, 100, 300, 500, 1000]  # 训练轮数节点
    current_stage_idx = 0
    
    print("=" * 60)
    print("Q-Learning 学习过程可视化")
    print("=" * 60)
    
    # 训练并保存快照
    snapshots = {}
    
    for episode in range(stages[-1] + 1):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        
        agent.update_epsilon()
        
        # 在关键节点保存Q表快照
        if episode in stages:
            snapshots[episode] = agent.q_table.copy()
            print(f"已保存第 {episode} 轮的Q表快照")
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Q-Learning 学习过程 - Q表演化", fontsize=16)
    
    for idx, (ax, stage) in enumerate(zip(axes.flat, stages)):
        q_table = snapshots[stage]
        
        # 显示每个状态的最大Q值（价值函数）
        v_values = np.max(q_table, axis=1).reshape(env.size, env.size)
        
        im = ax.imshow(v_values, cmap='RdYlGn', vmin=-10, vmax=10)
        ax.set_title(f"训练 {stage} 轮", fontsize=14)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        
        # 标记特殊位置
        ax.plot(env.goal_pos[1], env.goal_pos[0], 'r*', markersize=15)
        for trap in env.traps:
            ax.plot(trap[1], trap[0], 'rx', markersize=10)
        
        # 在每个格子中显示数值
        for i in range(env.size):
            for j in range(env.size):
                ax.text(j, i, f'{v_values[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=9)
    
    # 隐藏多余的子图
    if len(stages) < 6:
        for idx in range(len(stages), 6):
            axes.flat[idx].axis('off')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1)
    cbar.set_label('状态价值 V(s) = max(Q(s,a))', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('learning_process.png', dpi=150)
    print("\n学习过程图已保存为 'learning_process.png'")
    plt.show()
    
    # 绘制最终策略
    fig2 = visualize_policy_grid(snapshots[stages[-1]], env, 
                                  f"最终策略 (训练 {stages[-1]} 轮)")
    fig2.savefig('final_policy.png', dpi=150)
    print("最终策略图已保存为 'final_policy.png'")
    plt.show()
    
    # 绘制最终Q值热力图
    fig3 = visualize_q_values(snapshots[stages[-1]], env,
                              f"最终Q表 (训练 {stages[-1]} 轮)")
    fig3.savefig('final_q_values.png', dpi=150)
    print("最终Q值图已保存为 'final_q_values.png'")
    plt.show()


if __name__ == "__main__":
    demonstrate_learning_process()
