"""
可视化测试脚本 - 美观的图形界面展示
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.widgets import Button
import matplotlib
from grid_world_env import GridWorldEnv
from q_learning_agent import QLearningAgent

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class VisualTester:
    """可视化测试器"""
    
    def __init__(self, model_path="q_learning_model.pkl"):
        # 创建环境
        self.env = GridWorldEnv(size=4)
        
        # 创建智能体并加载模型
        self.agent = QLearningAgent(
            n_states=self.env.observation_space,
            n_actions=self.env.action_space
        )
        
        try:
            self.agent.load(model_path)
            self.has_model = True
        except:
            print(f"警告: 无法加载模型 {model_path}")
            self.has_model = False
        
        # 初始化状态
        self.reset_episode()
        
        # 创建图形
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Q-Learning 智能体测试', fontsize=20, fontweight='bold', y=0.98)
        
        # 创建子图
        self._create_layout()
        
        # 绘制初始状态
        self._draw_grid()
        self._draw_q_table()
        self._draw_info()
        
        # 添加按钮
        self._create_buttons()
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    def reset_episode(self):
        """重置回合"""
        self.state = self.env.reset()
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_history = []
        self.done = False
        self.path = [self.env.current_pos]
    
    def _create_layout(self):
        """创建布局"""
        # 主网格区域 (左)
        self.ax_grid = self.fig.add_subplot(2, 3, (1, 4))
        self.ax_grid.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title('网格世界 (Grid World)', fontsize=14, fontweight='bold', pad=10)
        
        # 关闭坐标轴
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # 绘制网格线
        for i in range(self.env.size + 1):
            self.ax_grid.axhline(y=i-0.5, color='gray', linewidth=1, alpha=0.3)
            self.ax_grid.axvline(x=i-0.5, color='gray', linewidth=1, alpha=0.3)
        
        # Q表区域 (右上)
        self.ax_qtable = self.fig.add_subplot(2, 3, 2)
        self.ax_qtable.set_title('Q值表 (Q-Table)', fontsize=14, fontweight='bold')
        
        # 信息面板 (右中)
        self.ax_info = self.fig.add_subplot(2, 3, 3)
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        self.ax_info.set_title('训练信息', fontsize=14, fontweight='bold')
        
        # 回合历史 (右下)
        self.ax_history = self.fig.add_subplot(2, 3, 6)
        self.ax_history.set_title('回合奖励历史', fontsize=14, fontweight='bold')
        self.ax_history.set_xlabel('回合')
        self.ax_history.set_ylabel('总奖励')
        self.episode_rewards = []
    
    def _draw_grid(self):
        """绘制网格世界"""
        self.ax_grid.clear()
        self.ax_grid.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        
        # 绘制网格背景
        for i in range(self.env.size):
            for j in range(self.env.size):
                pos = (i, j)
                
                # 根据位置设置颜色
                if pos == self.env.goal_pos:
                    color = '#86efac'  # 绿色 - 终点
                    label = 'G'
                    emoji = 'G'
                elif pos in self.env.traps:
                    color = '#fca5a5'  # 红色 - 陷阱
                    label = 'X'
                    emoji = 'X'
                elif pos == self.env.start_pos:
                    color = '#93c5fd'  # 蓝色 - 起点
                    label = 'S'
                    emoji = 'S'
                else:
                    color = '#f8fafc'  # 浅灰 - 普通格子
                    label = ''
                    emoji = ''
                
                # 绘制格子
                rect = FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    facecolor=color,
                    edgecolor='#64748b',
                    linewidth=2
                )
                self.ax_grid.add_patch(rect)
                
                # 添加标签
                if label:
                    self.ax_grid.text(j, i, emoji, ha='center', va='center', fontsize=20)
                
                # 如果有模型，显示最佳动作箭头
                if self.has_model and pos not in self.env.traps and pos != self.env.goal_pos:
                    state_idx = i * self.env.size + j
                    q_values = self.agent.q_table[state_idx]
                    best_action = np.argmax(q_values)
                    
                    # 动作箭头
                    arrows = {
                        0: (0, -0.25),    # 上
                        1: (0, 0.25),     # 下
                        2: (-0.25, 0),    # 左
                        3: (0.25, 0)      # 右
                    }
                    dx, dy = arrows[best_action]
                    
                    # 透明度根据Q值相对大小
                    q_max, q_min = q_values.max(), q_values.min()
                    if q_max > q_min:
                        intensity = (q_values[best_action] - q_min) / (q_max - q_min + 0.01)
                        alpha = 0.3 + 0.7 * intensity
                    else:
                        alpha = 0.5
                    
                    self.ax_grid.arrow(j, i, dx, dy, head_width=0.12, head_length=0.1,
                                     fc='#4f46e5', ec='#4f46e5', alpha=alpha, linewidth=2)
        
        # 绘制路径
        if len(self.path) > 1:
            path_y = [p[0] for p in self.path]
            path_x = [p[1] for p in self.path]
            self.ax_grid.plot(path_x, path_y, 'o-', color='#f59e0b', 
                            markersize=8, linewidth=2, alpha=0.6, zorder=5)
        
        # 绘制智能体
        row, col = self.env.current_pos
        agent_circle = Circle((col, row), 0.25, facecolor='#fbbf24', 
                             edgecolor='#b45309', linewidth=3, zorder=10)
        self.ax_grid.add_patch(agent_circle)
        self.ax_grid.text(col, row, 'A', ha='center', va='center', 
                         fontsize=18, fontweight='bold', color='#b45309', zorder=11)
        
        # 添加标题
        status = "[完成]" if self.done else "[进行中]"
        self.ax_grid.set_title(f'网格世界 - {status}', fontsize=14, fontweight='bold', pad=10)
    
    def _draw_q_table(self):
        """绘制Q表热力图"""
        self.ax_qtable.clear()
        
        if not self.has_model:
            self.ax_qtable.text(0.5, 0.5, '无模型数据\n请先训练', 
                               ha='center', va='center', fontsize=14)
            return
        
        # 获取当前状态的Q值
        q_values = self.agent.q_table[self.state]
        
        # 创建条形图
        actions = ['[上]', '[下]', '[左]', '[右]']
        colors = ['#3b82f6' if q > 0 else '#ef4444' for q in q_values]
        
        bars = self.ax_qtable.barh(actions, q_values, color=colors, alpha=0.7, edgecolor='black')
        
        # 在条形上显示数值
        for i, (bar, val) in enumerate(zip(bars, q_values)):
            width = bar.get_width()
            self.ax_qtable.text(width + 0.1 if width >= 0 else width - 0.1, 
                               bar.get_y() + bar.get_height()/2,
                               f'{val:.2f}',
                               ha='left' if width >= 0 else 'right',
                               va='center', fontsize=11, fontweight='bold')
        
        # 高亮最佳动作
        best_action = np.argmax(q_values)
        bars[best_action].set_edgecolor('#10b981')
        bars[best_action].set_linewidth(4)
        
        self.ax_qtable.set_xlim(-15, 15)
        self.ax_qtable.axvline(x=0, color='black', linewidth=1)
        self.ax_qtable.set_xlabel('Q值', fontsize=11)
        self.ax_qtable.set_title(f'状态 {self.state} 的Q值 (位置: {self.env.current_pos})', 
                                fontsize=12, fontweight='bold')
        self.ax_qtable.grid(True, alpha=0.3, axis='x')
    
    def _draw_info(self):
        """绘制信息面板"""
        self.ax_info.clear()
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        
        # 信息文本
        info_text = f"""
当前回合统计

状态 (State): {self.state}
位置 (Position): {self.env.current_pos}

当前奖励: {self.episode_reward:.1f}
当前步数: {self.episode_steps}

探索率 (ε): {self.agent.epsilon:.4f}
训练步数: {self.agent.training_step}
        """
        
        if self.done:
            pos = self.env.current_pos
            if pos == self.env.goal_pos:
                result = "[成功到达终点!]"
                color = '#10b981'
            elif pos in self.env.traps:
                result = "[踩中陷阱!]"
                color = '#ef4444'
            else:
                result = "[步数耗尽]"
                color = '#f59e0b'
            
            info_text += f"\n\n{'='*20}\n{result}"
        
        self.ax_info.text(0.5, 0.5, info_text, transform=self.ax_info.transAxes,
                         fontsize=12, verticalalignment='center',
                         horizontalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='#f8fafc', 
                                  edgecolor='#e2e8f0', linewidth=2, pad=1))
    
    def _draw_history(self):
        """绘制历史曲线"""
        self.ax_history.clear()
        
        if len(self.episode_rewards) > 0:
            episodes = range(1, len(self.episode_rewards) + 1)
            self.ax_history.plot(episodes, self.episode_rewards, 'o-', 
                               color='#4f46e5', linewidth=2, markersize=6)
            
            # 添加平均线
            avg_reward = np.mean(self.episode_rewards)
            self.ax_history.axhline(y=avg_reward, color='#ef4444', 
                                   linestyle='--', linewidth=2, label=f'平均: {avg_reward:.1f}')
            
            self.ax_history.set_xlabel('回合', fontsize=11)
            self.ax_history.set_ylabel('总奖励', fontsize=11)
            self.ax_history.set_title('回合奖励历史', fontsize=12, fontweight='bold')
            self.ax_history.legend()
            self.ax_history.grid(True, alpha=0.3)
    
    def _create_buttons(self):
        """创建控制按钮"""
        # 按钮位置
        ax_step = self.fig.add_axes([0.15, 0.02, 0.12, 0.05])
        ax_auto = self.fig.add_axes([0.30, 0.02, 0.12, 0.05])
        ax_reset = self.fig.add_axes([0.45, 0.02, 0.12, 0.05])
        ax_new = self.fig.add_axes([0.60, 0.02, 0.12, 0.05])
        
        # 创建按钮
        self.btn_step = Button(ax_step, '单步执行', color='#dbeafe', hovercolor='#bfdbfe')
        self.btn_auto = Button(ax_auto, '自动播放', color='#d1fae5', hovercolor='#a7f3d0')
        self.btn_reset = Button(ax_reset, '重置位置', color='#fef3c7', hovercolor='#fde68a')
        self.btn_new = Button(ax_new, '新回合', color='#f3e8ff', hovercolor='#e9d5ff')
        
        # 绑定事件
        self.btn_step.on_clicked(self._on_step)
        self.btn_auto.on_clicked(self._on_auto)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_new.on_clicked(self._on_new_episode)
    
    def _on_step(self, event):
        """单步执行"""
        if self.done:
            return
        
        # 选择动作
        action = self.agent.choose_action(self.state, training=False)
        
        # 执行动作
        next_state, reward, self.done, info = self.env.step(action)
        
        # 更新统计
        self.episode_reward += reward
        self.episode_steps += 1
        self.state = next_state
        self.path.append(self.env.current_pos)
        
        # 更新显示
        self._draw_grid()
        self._draw_q_table()
        self._draw_info()
        
        if self.done:
            self.episode_rewards.append(self.episode_reward)
            self._draw_history()
        
        self.fig.canvas.draw_idle()
    
    def _on_auto(self, event):
        """自动播放整个回合"""
        if self.done:
            self._on_new_episode(None)
        
        while not self.done:
            self._on_step(None)
            plt.pause(0.5)  # 短暂延迟以便观察
            self.fig.canvas.flush_events()
    
    def _on_reset(self, event):
        """重置到起点"""
        self.state = self.env.reset()
        self.path = [self.env.current_pos]
        self._draw_grid()
        self._draw_q_table()
        self._draw_info()
        self.fig.canvas.draw_idle()
    
    def _on_new_episode(self, event):
        """开始新回合"""
        if not self.done and self.episode_steps > 0:
            self.episode_rewards.append(self.episode_reward)
        
        self.reset_episode()
        self._draw_grid()
        self._draw_q_table()
        self._draw_info()
        self._draw_history()
        self.fig.canvas.draw_idle()
    
    def show(self):
        """显示界面"""
        plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("Q-Learning 可视化测试")
    print("=" * 60)
    print("\n功能说明:")
    print("  [单步执行] - 手动一步步执行动作")
    print("  [自动播放] - 自动执行完整回合")
    print("  [重置位置] - 回到起点")
    print("  [新回合]   - 开始新的测试回合")
    print("\n" + "=" * 60)
    
    tester = VisualTester("q_learning_model.pkl")
    tester.show()


if __name__ == "__main__":
    main()
