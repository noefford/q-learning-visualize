"""
可视化训练脚本 - 实时展示Q-Learning学习过程 (英文版，避免字体问题)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, TextBox
import matplotlib
from grid_world_env import GridWorldEnv
from q_learning_agent import QLearningAgent


class VisualTrainer:
    """Visual Q-Learning Trainer"""
    
    def __init__(self):
        # Create environment
        self.env = GridWorldEnv(size=4)
        
        # Create agent
        self.agent = QLearningAgent(
            n_states=self.env.observation_space,
            n_actions=self.env.action_space,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.is_training = False
        self.current_state = None
        self.last_update_info = None
        
        # History
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        
        # Create figure
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.suptitle('Q-Learning Real-time Training Visualization', 
                         fontsize=18, fontweight='bold', y=0.98)
        
        # Create layout
        self._create_layout()
        
        # Initialize
        self.reset_training()
        self._draw_all()
        
        # Create buttons
        self._create_buttons()
        
        # 调整布局，避免重叠 (bottom增加以适应更多按钮)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12, 
                                wspace=0.3, hspace=0.4)
    
    def reset_training(self):
        """Reset training"""
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
        """Create layout"""
        # 1. Q-table heatmap (top-left)
        self.ax_qtable = self.fig.add_subplot(3, 3, 1)
        self.ax_qtable.set_title('Q-Table State (Max Q-value)', fontsize=12, fontweight='bold')
        
        # 2. Update details (top-center)
        self.ax_detail = self.fig.add_subplot(3, 3, 2)
        self.ax_detail.set_xlim(0, 1)
        self.ax_detail.set_ylim(0, 1)
        self.ax_detail.axis('off')
        self.ax_detail.set_title('Current Update Detail', fontsize=12, fontweight='bold')
        
        # 3. Formula (top-right)
        self.ax_formula = self.fig.add_subplot(3, 3, 3)
        self.ax_formula.set_xlim(0, 1)
        self.ax_formula.set_ylim(0, 1)
        self.ax_formula.axis('off')
        self.ax_formula.set_title('Q-Learning Formula', fontsize=12, fontweight='bold')
        
        # 4. Grid world (middle-left)
        self.ax_grid = self.fig.add_subplot(3, 3, 4)
        self.ax_grid.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title('Grid World', fontsize=12, fontweight='bold')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # 5. Policy visualization (middle-center)
        self.ax_policy = self.fig.add_subplot(3, 3, 5)
        self.ax_policy.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_aspect('equal')
        self.ax_policy.invert_yaxis()
        self.ax_policy.set_title('Current Policy', fontsize=12, fontweight='bold')
        self.ax_policy.set_xticks([])
        self.ax_policy.set_yticks([])
        
        # 6. Training stats (middle-right)
        self.ax_stats = self.fig.add_subplot(3, 3, 6)
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Training Statistics', fontsize=11, fontweight='bold', pad=5)
        
        # 7. Reward curve (bottom-left)
        self.ax_reward = self.fig.add_subplot(3, 3, 7)
        self.ax_reward.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)
        
        # 8. Steps curve (bottom-center)
        self.ax_step = self.fig.add_subplot(3, 3, 8)
        self.ax_step.set_title('Episode Steps', fontsize=12, fontweight='bold')
        self.ax_step.set_xlabel('Episode')
        self.ax_step.set_ylabel('Steps')
        self.ax_step.grid(True, alpha=0.3)
        
        # 9. Epsilon curve (bottom-right)
        self.ax_epsilon = self.fig.add_subplot(3, 3, 9)
        self.ax_epsilon.set_title('Exploration Rate (Epsilon)', fontsize=12, fontweight='bold')
        self.ax_epsilon.set_xlabel('Episode')
        self.ax_epsilon.set_ylabel('Epsilon')
        self.ax_epsilon.grid(True, alpha=0.3)
    
    def _draw_qtable(self):
        """Draw Q-table as text grid (no heatmap to avoid overlap)"""
        self.ax_qtable.clear()
        self.ax_qtable.set_xlim(0, 1)
        self.ax_qtable.set_ylim(0, 1)
        self.ax_qtable.axis('off')
        self.ax_qtable.set_title('Q-Table (State:MaxQ)', fontsize=10, fontweight='bold', pad=5)
        
        # 创建文本表格显示Q值
        max_q = np.max(self.agent.q_table, axis=1).reshape(self.env.size, self.env.size)
        
        table_text = ""
        for i in range(self.env.size):
            row_str = ""
            for j in range(self.env.size):
                state = i * self.env.size + j
                pos = (i, j)
                q_val = max_q[i, j]
                
                # 标记特殊状态
                marker = ""
                if pos == self.env.goal_pos:
                    marker = "[G]"
                elif pos in self.env.traps:
                    marker = "[X]"
                elif state == self.current_state:
                    marker = "[*]"
                
                row_str += f"{state:2d}:{q_val:5.1f}{marker:4s}  "
            table_text += row_str + "\n"
        
        self.ax_qtable.text(0.5, 0.5, table_text, transform=self.ax_qtable.transAxes,
                           fontsize=8, verticalalignment='center',
                           horizontalalignment='center', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='#f8fafc', 
                                    edgecolor='#e2e8f0', linewidth=1, pad=0.5))
    
    def _draw_detail(self):
        """Draw update details"""
        self.ax_detail.clear()
        self.ax_detail.set_xlim(0, 1)
        self.ax_detail.set_ylim(0, 1)
        self.ax_detail.axis('off')
        self.ax_detail.set_title('Current Update Detail', fontsize=11, fontweight='bold', pad=5)
        
        if self.last_update_info is None:
            info = "Waiting to start training...\n\nClick [Start Training] button"
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
        
        action_names = ['Up', 'Down', 'Left', 'Right']
        
        detail_text = f"""State: {state} -> {next_state}
Action: {action_names[action]}
Reward: {reward:.2f}

Q-Value Update:
  Old: {old_q:.3f}
  New: {new_q:.3f}
  Change: {new_q - old_q:+.3f}

TD Target: {td_target:.3f}
TD Error: {td_error:+.3f}
        """
        
        self.ax_detail.text(0.5, 0.5, detail_text, transform=self.ax_detail.transAxes,
                           fontsize=9, verticalalignment='center',
                           horizontalalignment='center', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='#eff6ff', 
                                    edgecolor='#3b82f6', linewidth=1.5, pad=0.8))
    
    def _draw_formula(self):
        """Draw formula"""
        self.ax_formula.clear()
        self.ax_formula.set_xlim(0, 1)
        self.ax_formula.set_ylim(0, 1)
        self.ax_formula.axis('off')
        self.ax_formula.set_title('Q-Learning Formula', fontsize=11, fontweight='bold', pad=5)
        
        formula_text = "Q(s,a) = Q(s,a) + alpha * [r + gamma*maxQ(s',a') - Q(s,a)]"
        
        self.ax_formula.text(0.5, 0.85, formula_text, transform=self.ax_formula.transAxes,
                            fontsize=11, verticalalignment='top',
                            horizontalalignment='center', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='#f0fdf4', 
                                     edgecolor='#22c55e', linewidth=2, pad=1))
        
        if self.last_update_info is not None:
            info = self.last_update_info
            alpha = self.agent.lr
            gamma = self.agent.gamma
            
            calc_text = f"""Parameters:
alpha (learning rate) = {alpha}
gamma (discount) = {gamma}

Calculation:
TD Target = r + gamma*maxQ(s')
         = {info['reward']:.2f} + {gamma}*{info['max_next_q']:.3f}
         = {info['td_target']:.3f}

TD Error = {info['td_target']:.3f} - {info['old_q']:.3f}
        = {info['td_error']:+.3f}

Q Update = {info['old_q']:.3f} + {alpha}*{info['td_error']:+.3f}
        = {info['new_q']:.3f}
            """
            
            self.ax_formula.text(0.5, 0.42, calc_text, transform=self.ax_formula.transAxes,
                                fontsize=8, verticalalignment='top',
                                horizontalalignment='center', family='monospace',
                                bbox=dict(boxstyle='round', facecolor='#eff6ff', 
                                         edgecolor='#3b82f6', linewidth=1, pad=0.6))
    
    def _draw_grid(self):
        """Draw grid world"""
        self.ax_grid.clear()
        self.ax_grid.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title('Grid World', fontsize=12, fontweight='bold')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                pos = (i, j)
                state = i * self.env.size + j
                
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
                
                rect = FancyBboxPatch(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    facecolor=color,
                    edgecolor='#64748b',
                    linewidth=2
                )
                self.ax_grid.add_patch(rect)
                
                if label:
                    self.ax_grid.text(j, i, label, ha='center', va='center', 
                                     fontsize=16, fontweight='bold')
                
                self.ax_grid.text(j, i+0.3, f'{state}', ha='center', va='center', 
                                 fontsize=8, color='#64748b')
                
                if state == self.current_state:
                    circle = plt.Circle((j, i), 0.2, color='#fbbf24', 
                                       ec='#b45309', linewidth=2)
                    self.ax_grid.add_patch(circle)
                    self.ax_grid.text(j, i, 'A', ha='center', va='center', 
                                     fontsize=12, fontweight='bold', color='#b45309')
    
    def _draw_policy(self):
        """Draw policy"""
        self.ax_policy.clear()
        self.ax_policy.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_aspect('equal')
        self.ax_policy.invert_yaxis()
        self.ax_policy.set_title('Current Policy (Best Action)', fontsize=12, fontweight='bold')
        self.ax_policy.set_xticks([])
        self.ax_policy.set_yticks([])
        
        arrows = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
        arrow_symbols = {0: '^', 1: 'v', 2: '<', 3: '>'}
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                pos = (i, j)
                state = i * self.env.size + j
                
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
                
                if state == self.current_state:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                    edgecolor='blue', linewidth=3)
                    self.ax_policy.add_patch(rect)
    
    def _draw_stats(self):
        """Draw statistics"""
        self.ax_stats.clear()
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Training Statistics', fontsize=11, fontweight='bold', pad=5)
        
        status = "TRAINING" if self.is_training else "PAUSED"
        
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        avg_steps = np.mean(self.episode_steps[-10:]) if self.episode_steps else 0
        
        # 简化文本，减小字号
        stats_text = f"""Status: {status}
Ep: {self.episode} | Step: {self.episode_step}
Reward: {self.episode_reward:.1f}
Total: {self.total_steps}
e: {self.agent.epsilon:.3f} | a: {self.agent.lr} | g: {self.agent.gamma}
AvgR: {avg_reward:.1f} | AvgS: {avg_steps:.1f}
        """
        
        self.ax_stats.text(0.5, 0.5, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=9, verticalalignment='center',
                          horizontalalignment='center', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='#fefce8', 
                                   edgecolor='#eab308', linewidth=1.5, pad=0.8))
    
    def _draw_curves(self):
        """Draw training curves"""
        self.ax_reward.clear()
        if len(self.episode_rewards) > 0:
            self.ax_reward.plot(self.episode_rewards, alpha=0.3, color='blue')
            if len(self.episode_rewards) >= 10:
                smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                self.ax_reward.plot(range(9, len(self.episode_rewards)), smoothed, 
                                   color='red', linewidth=2, label='10-ep avg')
            self.ax_reward.legend()
        self.ax_reward.set_title('Episode Rewards', fontsize=11, fontweight='bold')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)
        
        self.ax_step.clear()
        if len(self.episode_steps) > 0:
            self.ax_step.plot(self.episode_steps, alpha=0.3, color='green')
            if len(self.episode_steps) >= 10:
                smoothed = np.convolve(self.episode_steps, np.ones(10)/10, mode='valid')
                self.ax_step.plot(range(9, len(self.episode_steps)), smoothed, 
                                 color='red', linewidth=2)
        self.ax_step.set_title('Episode Steps', fontsize=11, fontweight='bold')
        self.ax_step.set_xlabel('Episode')
        self.ax_step.set_ylabel('Steps')
        self.ax_step.grid(True, alpha=0.3)
        
        self.ax_epsilon.clear()
        if len(self.epsilon_history) > 0:
            self.ax_epsilon.plot(self.epsilon_history, color='purple', linewidth=2)
        self.ax_epsilon.set_title('Exploration Rate', fontsize=11, fontweight='bold')
        self.ax_epsilon.set_xlabel('Episode')
        self.ax_epsilon.set_ylabel('Epsilon')
        self.ax_epsilon.grid(True, alpha=0.3)
    
    def _draw_all(self):
        """Draw all components"""
        self._draw_qtable()
        self._draw_detail()
        self._draw_formula()
        self._draw_grid()
        self._draw_policy()
        self._draw_stats()
        self._draw_curves()
        self.fig.canvas.draw_idle()
    
    def _create_buttons(self):
        """Create control buttons"""
        # 第一行按钮
        ax_start = self.fig.add_axes([0.08, 0.01, 0.10, 0.045])
        ax_pause = self.fig.add_axes([0.20, 0.01, 0.10, 0.045])
        ax_step = self.fig.add_axes([0.32, 0.01, 0.10, 0.045])
        ax_reset = self.fig.add_axes([0.44, 0.01, 0.10, 0.045])
        
        self.btn_start = Button(ax_start, 'Start', color='#22c55e', hovercolor='#16a34a')
        self.btn_pause = Button(ax_pause, 'Pause', color='#eab308', hovercolor='#ca8a04')
        self.btn_step = Button(ax_step, 'Step', color='#3b82f6', hovercolor='#2563eb')
        self.btn_reset = Button(ax_reset, 'Reset', color='#ef4444', hovercolor='#dc2626')
        
        self.btn_start.on_clicked(self._on_start)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_step.on_clicked(self._on_step)
        self.btn_reset.on_clicked(self._on_reset)
        
        # 批量训练输入框和按钮
        ax_text = self.fig.add_axes([0.58, 0.01, 0.08, 0.045])
        ax_batch = self.fig.add_axes([0.68, 0.01, 0.12, 0.045])
        
        self.text_batch = TextBox(ax_text, '', initial='10')
        self.text_batch.label.set_visible(False)
        
        self.btn_batch = Button(ax_batch, 'Train N Episodes', color='#8b5cf6', hovercolor='#7c3aed')
        self.btn_batch.on_clicked(self._on_batch_train)
    
    def _train_one_step(self):
        """Train one step"""
        if self.current_state is None:
            self.current_state = self.env.reset()
            self.episode_reward = 0
            self.episode_step = 0
        
        action = self.agent.choose_action(self.current_state, training=True)
        next_state, reward, done, info = self.env.step(action)
        
        old_q = self.agent.q_table[self.current_state, action]
        
        if done:
            td_target = reward
        else:
            max_next_q = np.max(self.agent.q_table[next_state])
            td_target = reward + self.agent.gamma * max_next_q
        
        self.agent.learn(self.current_state, action, reward, next_state, done)
        new_q = self.agent.q_table[self.current_state, action]
        
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
        
        self.episode_reward += reward
        self.episode_step += 1
        self.total_steps += 1
        self.current_state = next_state
        
        if done:
            self.agent.update_epsilon()
            self.episode_rewards.append(self.episode_reward)
            self.episode_steps.append(self.episode_step)
            self.epsilon_history.append(self.agent.epsilon)
            self.episode += 1
            
            self.current_state = self.env.reset()
            self.episode_reward = 0
            self.episode_step = 0
        
        return True
    
    def _on_start(self, event):
        """Start training"""
        if not self.is_training:
            self.is_training = True
            self._train_loop()
    
    def _on_pause(self, event):
        """Pause training"""
        self.is_training = False
        self._draw_all()
    
    def _on_step(self, event):
        """Step training"""
        self.is_training = False
        self._train_one_step()
        self._draw_all()
    
    def _on_reset(self, event):
        """Reset training"""
        self.is_training = False
        self.reset_training()
        self.current_state = self.env.reset()
        self._draw_all()
    
    def _on_batch_train(self, event):
        """Train N episodes at once"""
        try:
            n_episodes = int(self.text_batch.text)
            if n_episodes <= 0:
                print("Please enter a positive number")
                return
            if n_episodes > 1000:
                print("Maximum 1000 episodes at once, using 1000")
                n_episodes = 1000
            
            print(f"Starting batch training: {n_episodes} episodes...")
            self._batch_train(n_episodes)
        except ValueError:
            print(f"Invalid input: '{self.text_batch.text}', please enter a number")
    
    def _batch_train(self, n_episodes):
        """Batch train for N episodes"""
        self.is_training = True
        
        for ep in range(n_episodes):
            if not self.is_training:
                break
            
            # Reset for new episode
            self.current_state = self.env.reset()
            self.episode_reward = 0
            self.episode_step = 0
            
            # Run one episode
            done = False
            step_count = 0
            while not done and step_count < 100:
                action = self.agent.choose_action(self.current_state, training=True)
                next_state, reward, done, info = self.env.step(action)
                self.agent.learn(self.current_state, action, reward, next_state, done)
                
                self.episode_reward += reward
                self.episode_step += 1
                self.total_steps += 1
                self.current_state = next_state
                step_count += 1
            
            # Episode finished
            self.agent.update_epsilon()
            self.episode_rewards.append(self.episode_reward)
            self.episode_steps.append(self.episode_step)
            self.epsilon_history.append(self.agent.epsilon)
            self.episode += 1
            
            # Update display every 5 episodes
            if ep % 5 == 0 or ep == n_episodes - 1:
                self._draw_all()
                plt.pause(0.01)
                self.fig.canvas.flush_events()
        
        self.is_training = False
        self._draw_all()
        print(f"Batch training completed: {n_episodes} episodes")
    
    def _train_loop(self):
        """Training loop"""
        if self.is_training:
            self._train_one_step()
            self._draw_all()
            plt.pause(0.1)
            self.fig.canvas.flush_events()
            
            if self.is_training:
                plt.gcf().canvas.get_tk_widget().after(10, self._train_loop)
    
    def show(self):
        """Show interface"""
        plt.show()


def main():
    """Main function"""
    print("=" * 60)
    print("Q-Learning Visual Training")
    print("=" * 60)
    print("\nControls:")
    print("  [Start] - Auto training (continuous)")
    print("  [Pause] - Pause training")
    print("  [Step]  - Manual single step")
    print("  [Reset] - Reset all training")
    print("  [TextBox + Train N Episodes] - Train N episodes at once")
    print("\nUsage:")
    print("  1. Enter a number in the text box (e.g., 100)")
    print("  2. Click 'Train N Episodes' to train quickly")
    print("\nPanels:")
    print("  - Q-Table: Q-values for each state")
    print("  - Update Detail: Current step's Q-value update")
    print("  - Formula: Q-Learning formula with calculation")
    print("  - Grid World: Current agent position")
    print("  - Policy: Best action for each state")
    print("=" * 60)
    
    trainer = VisualTrainer()
    trainer.show()


if __name__ == "__main__":
    main()
