"""
可视化训练脚本 - 实时展示Q-Learning学习过程
包含Q表更新、公式展示、训练曲线等
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import Button
import matplotlib
from grid_world_env import GridWorldEnv
from q_learning_agent import QLearningAgent

# 设置中文字体 - 尝试多种方案
import matplotlib.font_manager as fm

# 尝试查找系统中的中文字体
def find_chinese_font():
    """查找可用的中文字体"""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun',
        'FangSong', 'KaiTi', 'Arial Unicode MS',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'Source Han Sans CN',
        'Hiragino Sans GB', 'STHeiti', 'STSong'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            print(f"使用中文字体: {font}")
            return font
    
    return None

chinese_font = find_chinese_font()

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
else:
    # 如果没有找到中文字体，使用默认字体并显示警告
    print("警告: 未找到中文字体，尝试使用默认字体")
    print("建议安装: pip install matplotlib-fonts-cn 或手动安装中文字体")

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class VisualTrainer:
    """可视化训练器"""
    
    def __init__(self):
        # 创建环境
        self.env = GridWorldEnv(size=4)
        
        # 创建智能体
        self.agent = QLearningAgent(
            n_states=self.env.observation_space,
            n_actions=self.env.action_space,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        # 训练状态
        self.episode = 0
        self.total_steps = 0
        self.is_training = False
        self.current_state = None
        self.last_update_info = None
        
        # 历史数据
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        
        # 创建图形
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.suptitle('Q-Learning 实时训练可视化', fontsize=18, fontweight='bold', y=0.98)
        
        # 创建子图布局
        self._create_layout()
        
        # 初始化显示
        self.reset_training()
        self._draw_all()
        
        # 创建按钮
        self._create_buttons()
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    def reset_training(self):
        """重置训练"""
        self.env = GridWorldEnv(size=4)
        self.agent = QLearningAgent(
            n_states=self.env.observation_space,
            n_actions=self.env.action_space,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        self.episode = 0
        self.total_steps = 0
        self.is_training = False
        self.current_state = self.env.reset()
        self.episode_reward = 0
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.last_update_info = None
    
    def _create_layout(self):
        """创建布局"""
        # 1. Q表热力图 (左上)
        self.ax_qtable = self.fig.add_subplot(3, 3, 1)
        self.ax_qtable.set_title('Q表状态 (Q-Table)', fontsize=12, fontweight='bold')
        
        # 2. 当前更新详情 (中上)
        self.ax_detail = self.fig.add_subplot(3, 3, 2)
        self.ax_detail.set_xlim(0, 1)
        self.ax_detail.set_ylim(0, 1)
        self.ax_detail.axis('off')
        self.ax_detail.set_title('当前更新详情', fontsize=12, fontweight='bold')
        
        # 3. 公式展示 (右上)
        self.ax_formula = self.fig.add_subplot(3, 3, 3)
        self.ax_formula.set_xlim(0, 1)
        self.ax_formula.set_ylim(0, 1)
        self.ax_formula.axis('off')
        self.ax_formula.set_title('Q-Learning 更新公式', fontsize=12, fontweight='bold')
        
        # 4. 网格世界 (左中)
        self.ax_grid = self.fig.add_subplot(3, 3, 4)
        self.ax_grid.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title('网格世界', fontsize=12, fontweight='bold')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # 5. 策略可视化 (中中)
        self.ax_policy = self.fig.add_subplot(3, 3, 5)
        self.ax_policy.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_aspect('equal')
        self.ax_policy.invert_yaxis()
        self.ax_policy.set_title('当前策略 (Policy)', fontsize=12, fontweight='bold')
        self.ax_policy.set_xticks([])
        self.ax_policy.set_yticks([])
        
        # 6. 训练统计 (右中)
        self.ax_stats = self.fig.add_subplot(3, 3, 6)
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('训练统计', fontsize=12, fontweight='bold')
        
        # 7. 奖励曲线 (左下)
        self.ax_reward = self.fig.add_subplot(3, 3, 7)
        self.ax_reward.set_title('每轮奖励', fontsize=12, fontweight='bold')
        self.ax_reward.set_xlabel('回合')
        self.ax_reward.set_ylabel('奖励')
        self.ax_reward.grid(True, alpha=0.3)
        
        # 8. 步数曲线 (中下)
        self.ax_step = self.fig.add_subplot(3, 3, 8)
        self.ax_step.set_title('每轮步数', fontsize=12, fontweight='bold')
        self.ax_step.set_xlabel('回合')
        self.ax_step.set_ylabel('步数')
        self.ax_step.grid(True, alpha=0.3)
        
        # 9. 探索率曲线 (右下)
        self.ax_epsilon = self.fig.add_subplot(3, 3, 9)
        self.ax_epsilon.set_title('探索率衰减', fontsize=12, fontweight='bold')
        self.ax_epsilon.set_xlabel('回合')
        self.ax_epsilon.set_ylabel('Epsilon')
        self.ax_epsilon.grid(True, alpha=0.3)
    
    def _draw_qtable(self):
        """绘制Q表热力图"""
        self.ax_qtable.clear()
        self.ax_qtable.set_title('Q表状态 (Q-Table) - 每个格子的最大Q值', fontsize=11, fontweight='bold')
        
        # 计算每个状态的最大Q值
        max_q = np.max(self.agent.q_table, axis=1).reshape(self.env.size, self.env.size)
        
        # 绘制热力图
        im = self.ax_qtable.imshow(max_q, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        # 添加数值
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = i * self.env.size + j
                pos = (i, j)
                
                # 显示最大Q值
                text_color = 'white' if abs(max_q[i, j]) > 5 else 'black'
                self.ax_qtable.text(j, i, f'{max_q[i, j]:.1f}',
                                   ha='center', va='center', fontsize=9, color=text_color)
                
                # 高亮当前状态
                if state == self.current_state:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                    edgecolor='blue', linewidth=4)
                    self.ax_qtable.add_patch(rect)
                
                # 标记特殊位置
                if pos == self.env.goal_pos:
                    self.ax_qtable.plot(j, i, 'r*', markersize=15)
                elif pos in self.env.traps:
                    self.ax_qtable.plot(j, i, 'rx', markersize=12)
        
        self.ax_qtable.set_xticks(range(self.env.size))
        self.ax_qtable.set_yticks(range(self.env.size))
        plt.colorbar(im, ax=self.ax_qtable, fraction=0.046, pad=0.04)
    
    def _draw_detail(self):
        """绘制更新详情"""
        self.ax_detail.clear()
        self.ax_detail.set_xlim(0, 1)
        self.ax_detail.set_ylim(0, 1)
        self.ax_detail.axis('off')
        self.ax_detail.set_title('当前更新详情', fontsize=12, fontweight='bold')
        
        if self.last_update_info is None:
            info = "等待开始训练...\n\n点击 [开始训练] 按钮"
            self.ax_detail.text(0.5, 0.5, info, transform=self.ax_detail.transAxes,
                               fontsize=11, verticalalignment='center',
                               horizontalalignment='center',
                               bbox=dict(boxstyle='round', facecolor='#f8fafc', 
                                        edgecolor='#e2e8f0', linewidth=2, pad=1))
            return
        
        info = self.last_update_info
        state = info['state']
        action = info['action']
        reward = info['reward']
        next_state = info['next_state']
        old_q = info['old_q']
        new_q = info['new_q']
        td_target = info['td_target']
        td_error = info['td_error']
        
        action_names = ['上', '下', '左', '右']
        
        detail_text = f"""状态: {state} -> {next_state}
动作: {action_names[action]}
即时奖励: {reward:.2f}

Q值更新:
  旧值: {old_q:.3f}
  新值: {new_q:.3f}
  变化: {new_q - old_q:+.3f}

TD目标: {td_target:.3f}
TD误差: {td_error:+.3f}
        """
        
        self.ax_detail.text(0.5, 0.5, detail_text, transform=self.ax_detail.transAxes,
                           fontsize=10, verticalalignment='center',
                           horizontalalignment='center', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='#eff6ff', 
                                    edgecolor='#3b82f6', linewidth=2, pad=1))
    
    def _draw_formula(self):
        """绘制公式"""
        self.ax_formula.clear()
        self.ax_formula.set_xlim(0, 1)
        self.ax_formula.set_ylim(0, 1)
        self.ax_formula.axis('off')
        self.ax_formula.set_title('Q-Learning 更新公式', fontsize=12, fontweight='bold')
        
        # 基础公式
        formula_text = "Q(s,a) = Q(s,a) + α × [r + γ×maxQ(s',a') - Q(s,a)]"
        
        self.ax_formula.text(0.5, 0.85, formula_text, transform=self.ax_formula.transAxes,
                            fontsize=11, verticalalignment='top',
                            horizontalalignment='center', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='#f0fdf4', 
                                     edgecolor='#22c55e', linewidth=2, pad=1))
        
        # 如果有更新信息，显示具体数值
        if self.last_update_info is not None:
            info = self.last_update_info
            
            # 分解公式
            alpha = self.agent.lr
            gamma = self.agent.gamma
            
            calc_text = f"""参数值:
α (学习率) = {alpha}
γ (折扣因子) = {gamma}

计算过程:
TD目标 = r + γ×maxQ(s')
      = {info['reward']:.2f} + {gamma}×{info['max_next_q']:.3f}
      = {info['td_target']:.3f}

TD误差 = {info['td_target']:.3f} - {info['old_q']:.3f}
      = {info['td_error']:+.3f}

Q值更新 = {info['old_q']:.3f} + {alpha}×{info['td_error']:+.3f}
        = {info['new_q']:.3f}
            """
            
            self.ax_formula.text(0.5, 0.45, calc_text, transform=self.ax_formula.transAxes,
                                fontsize=9, verticalalignment='top',
                                horizontalalignment='center', family='monospace',
                                bbox=dict(boxstyle='round', facecolor='#eff6ff', 
                                         edgecolor='#3b82f6', linewidth=1, pad=0.8))
    
    def _draw_grid(self):
        """绘制网格世界"""
        self.ax_grid.clear()
        self.ax_grid.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title('网格世界', fontsize=12, fontweight='bold')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # 绘制网格
        for i in range(self.env.size):
            for j in range(self.env.size):
                pos = (i, j)
                state = i * self.env.size + j
                
                # 设置颜色
                if pos == self.env.goal_pos:
                    color = '#86efac'
                    label = 'G'
                elif pos in self.env.traps:
                    color = '#fca5a5'
                    label = 'X'
                elif pos == self.env.start_pos:
                    color = '#93c5fd'
                    label = 'S'
                else:
                    color = '#f8fafc'
                    label = ''
                
                # 绘制格子
                rect = FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    facecolor=color,
                    edgecolor='#64748b',
                    linewidth=2
                )
                self.ax_grid.add_patch(rect)
                
                if label:
                    self.ax_grid.text(j, i, label, ha='center', va='center', fontsize=16, fontweight='bold')
                
                # 显示状态编号
                self.ax_grid.text(j, i+0.3, f'{state}', ha='center', va='center', 
                                 fontsize=8, color='#64748b')
                
                # 高亮当前状态
                if state == self.current_state:
                    circle = plt.Circle((j, i), 0.2, color='#fbbf24', ec='#b45309', linewidth=2)
                    self.ax_grid.add_patch(circle)
                    self.ax_grid.text(j, i, 'A', ha='center', va='center', 
                                     fontsize=12, fontweight='bold', color='#b45309')
    
    def _draw_policy(self):
        """绘制策略"""
        self.ax_policy.clear()
        self.ax_policy.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_aspect('equal')
        self.ax_policy.invert_yaxis()
        self.ax_policy.set_title('当前策略 (最佳动作)', fontsize=12, fontweight='bold')
        self.ax_policy.set_xticks([])
        self.ax_policy.set_yticks([])
        
        # 动作箭头
        arrows = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
        arrow_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                pos = (i, j)
                state = i * self.env.size + j
                
                # 设置颜色
                if pos == self.env.goal_pos:
                    color = '#86efac'
                    symbol = 'G'
                elif pos in self.env.traps:
                    color = '#fca5a5'
                    symbol = 'X'
                else:
                    color = 'white'
                    q_values = self.agent.q_table[state]
                    best_action = np.argmax(q_values)
                    symbol = arrow_symbols[best_action]
                
                # 绘制格子
                rect = FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    facecolor=color,
                    edgecolor='#64748b',
                    linewidth=1
                )
                self.ax_policy.add_patch(rect)
                
                self.ax_policy.text(j, i, symbol, ha='center', va='center', 
                                   fontsize=20, fontweight='bold')
                
                # 高亮当前状态
                if state == self.current_state:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                    edgecolor='blue', linewidth=3)
                    self.ax_policy.add_patch(rect)
    
    def _draw_stats(self):
        """绘制统计信息"""
        self.ax_stats.clear()
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('训练统计', fontsize=12, fontweight='bold')
        
        status = "训练中..." if self.is_training else "已暂停"
        
        stats_text = f"""训练状态: {status}

当前回合: {self.episode}
当前步数: {self.episode_step}
回合奖励: {self.episode_reward:.1f}
总步数: {self.total_steps}

探索率 ε: {self.agent.epsilon:.4f}
学习率 α: {self.agent.lr}
折扣因子 γ: {self.agent.gamma}

平均奖励: {np.mean(self.episode_rewards[-10:]):.1f} (近10轮)
平均步数: {np.mean(self.episode_steps[-10:]):.1f} (近10轮)
        """
        
        self.ax_stats.text(0.5, 0.5, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='center',
                          horizontalalignment='center', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='#fefce8', 
                                   edgecolor='#eab308', linewidth=2, pad=1))
    
    def _draw_curves(self):
        """绘制训练曲线"""
        # 奖励曲线
        self.ax_reward.clear()
        if len(self.episode_rewards) > 0:
            self.ax_reward.plot(self.episode_rewards, alpha=0.3, color='blue')
            if len(self.episode_rewards) >= 10:
                smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                self.ax_reward.plot(range(9, len(self.episode_rewards)), smoothed, 
                                   color='red', linewidth=2, label='10轮平均')
            self.ax_reward.legend()
        self.ax_reward.set_title('每轮奖励', fontsize=11, fontweight='bold')
        self.ax_reward.set_xlabel('回合')
        self.ax_reward.set_ylabel('奖励')
        self.ax_reward.grid(True, alpha=0.3)
        
        # 步数曲线
        self.ax_step.clear()
        if len(self.episode_steps) > 0:
            self.ax_step.plot(self.episode_steps, alpha=0.3, color='green')
            if len(self.episode_steps) >= 10:
                smoothed = np.convolve(self.episode_steps, np.ones(10)/10, mode='valid')
                self.ax_step.plot(range(9, len(self.episode_steps)), smoothed, 
                                 color='red', linewidth=2)
        self.ax_step.set_title('每轮步数', fontsize=11, fontweight='bold')
        self.ax_step.set_xlabel('回合')
        self.ax_step.set_ylabel('步数')
        self.ax_step.grid(True, alpha=0.3)
        
        # 探索率曲线
        self.ax_epsilon.clear()
        if len(self.epsilon_history) > 0:
            self.ax_epsilon.plot(self.epsilon_history, color='purple', linewidth=2)
        self.ax_epsilon.set_title('探索率衰减', fontsize=11, fontweight='bold')
        self.ax_epsilon.set_xlabel('回合')
        self.ax_epsilon.set_ylabel('Epsilon')
        self.ax_epsilon.grid(True, alpha=0.3)
    
    def _draw_all(self):
        """绘制所有组件"""
        self._draw_qtable()
        self._draw_detail()
        self._draw_formula()
        self._draw_grid()
        self._draw_policy()
        self._draw_stats()
        self._draw_curves()
        self.fig.canvas.draw_idle()
    
    def _create_buttons(self):
        """创建控制按钮"""
        # 按钮位置
        ax_start = self.fig.add_axes([0.20, 0.02, 0.12, 0.04])
        ax_pause = self.fig.add_axes([0.35, 0.02, 0.12, 0.04])
        ax_step = self.fig.add_axes([0.50, 0.02, 0.12, 0.04])
        ax_reset = self.fig.add_axes([0.65, 0.02, 0.12, 0.04])
        
        # 创建按钮
        self.btn_start = Button(ax_start, '开始训练', color='#22c55e', hovercolor='#16a34a')
        self.btn_pause = Button(ax_pause, '暂停', color='#eab308', hovercolor='#ca8a04')
        self.btn_step = Button(ax_step, '单步', color='#3b82f6', hovercolor='#2563eb')
        self.btn_reset = Button(ax_reset, '重置', color='#ef4444', hovercolor='#dc2626')
        
        # 绑定事件
        self.btn_start.on_clicked(self._on_start)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_step.on_clicked(self._on_step)
        self.btn_reset.on_clicked(self._on_reset)
    
    def _train_one_step(self):
        """训练一步"""
        if self.current_state is None:
            self.current_state = self.env.reset()
            self.episode_reward = 0
            self.episode_step = 0
        
        # 选择动作
        action = self.agent.choose_action(self.current_state, training=True)
        
        # 执行动作
        next_state, reward, done, info = self.env.step(action)
        
        # 记录旧的Q值
        old_q = self.agent.q_table[self.current_state, action]
        
        # 计算TD目标
        if done:
            td_target = reward
        else:
            max_next_q = np.max(self.agent.q_table[next_state])
            td_target = reward + self.agent.gamma * max_next_q
        
        # 更新Q值
        self.agent.learn(self.current_state, action, reward, next_state, done)
        new_q = self.agent.q_table[self.current_state, action]
        
        # 记录更新信息
        self.last_update_info = {
            'state': self.current_state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'old_q': old_q,
            'new_q': new_q,
            'td_target': td_target,
            'td_error': td_target - old_q,
            'max_next_q': np.max(self.agent.q_table[next_state]) if not done else 0
        }
        
        # 更新统计
        self.episode_reward += reward
        self.episode_step += 1
        self.total_steps += 1
        self.current_state = next_state
        
        # 回合结束
        if done:
            self.agent.update_epsilon()
            self.episode_rewards.append(self.episode_reward)
            self.episode_steps.append(self.episode_step)
            self.epsilon_history.append(self.agent.epsilon)
            self.episode += 1
            
            # 开始新回合
            self.current_state = self.env.reset()
            self.episode_reward = 0
            self.episode_step = 0
        
        return True
    
    def _on_start(self, event):
        """开始训练"""
        if not self.is_training:
            self.is_training = True
            self._train_loop()
    
    def _on_pause(self, event):
        """暂停训练"""
        self.is_training = False
        self._draw_all()
    
    def _on_step(self, event):
        """单步训练"""
        self.is_training = False
        self._train_one_step()
        self._draw_all()
    
    def _on_reset(self, event):
        """重置训练"""
        self.is_training = False
        self.reset_training()
        self.current_state = self.env.reset()
        self._draw_all()
    
    def _train_loop(self):
        """训练循环"""
        if self.is_training:
            self._train_one_step()
            self._draw_all()
            plt.pause(0.1)  # 短暂延迟以便观察
            self.fig.canvas.flush_events()
            
            # 继续下一轮
            if self.is_training:
                plt.gcf().canvas.get_tk_widget().after(10, self._train_loop)
    
    def show(self):
        """显示界面"""
        plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("Q-Learning 可视化训练")
    print("=" * 60)
    print("\n功能说明:")
    print("  [开始训练] - 自动持续训练")
    print("  [暂停]     - 暂停训练")
    print("  [单步]     - 手动执行一步训练")
    print("  [重置]     - 重置所有训练状态")
    print("\n界面说明:")
    print("  - Q表状态: 每个格子的颜色表示最大Q值")
    print("  - 更新详情: 显示当前Q值的具体更新")
    print("  - 公式展示: Q-Learning公式及计算过程")
    print("  - 网格世界: 当前智能体位置")
    print("  - 当前策略: 每个格子的最佳动作")
    print("=" * 60)
    
    trainer = VisualTrainer()
    trainer.show()


if __name__ == "__main__":
    main()
