"""
自定义网格世界环境 (Grid World Environment)
一个简化版的强化学习环境，用于理解RL基础概念
"""
import numpy as np
from typing import Tuple, Optional


class GridWorldEnv:
    """
    网格世界环境：智能体需要从起点移动到终点，避开陷阱
    
    环境设置:
    - 4x4 的网格
    - 起点: 左上角 (0, 0)
    - 终点: 右下角 (3, 3)，到达获得奖励 +10
    - 陷阱: (1, 1) 和 (2, 2)，踩中惩罚 -10
    - 每一步移动: 惩罚 -1（鼓励智能体尽快到达终点）
    """
    
    def __init__(self, size: int = 4):
        self.size = size  # 网格大小
        self.start_pos = (0, 0)  # 起点
        self.goal_pos = (size - 1, size - 1)  # 终点
        
        # 定义陷阱位置
        self.traps = [(1, 1), (2, 2)]
        
        # 定义动作空间: 0=上, 1=下, 2=左, 3=右
        self.action_space = 4
        
        # 定义观测空间大小（网格中的位置总数）
        self.observation_space = size * size
        
        # 当前位置
        self.current_pos = self.start_pos
        
        # 记录步数
        self.steps = 0
        self.max_steps = 100  # 最大步数限制
        
    def reset(self) -> int:
        """
        重置环境，返回初始状态
        
        Returns:
            初始状态的索引 (0 到 size*size-1)
        """
        self.current_pos = self.start_pos
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> int:
        """
        将二维坐标转换为一维状态索引
        
        例如: 4x4网格中，(row, col) -> row * 4 + col
        (0,0) -> 0, (0,1) -> 1, ..., (3,3) -> 15
        """
        return self.current_pos[0] * self.size + self.current_pos[1]
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        执行一个动作，返回新的状态、奖励、是否结束、额外信息
        
        Args:
            action: 动作 (0=上, 1=下, 2=左, 3=右)
        
        Returns:
            state: 新状态的索引
            reward: 奖励值
            done: 是否结束
            info: 额外信息字典
        """
        self.steps += 1
        row, col = self.current_pos
        
        # 根据动作移动位置
        if action == 0:    # 上
            row = max(0, row - 1)
        elif action == 1:  # 下
            row = min(self.size - 1, row + 1)
        elif action == 2:  # 左
            col = max(0, col - 1)
        elif action == 3:  # 右
            col = min(self.size - 1, col + 1)
        
        self.current_pos = (row, col)
        
        # 计算奖励
        reward = -1  # 每步基础惩罚
        done = False
        
        # 检查是否到达终点
        if self.current_pos == self.goal_pos:
            reward = 10  # 到达终点获得正奖励
            done = True
        # 检查是否踩中陷阱
        elif self.current_pos in self.traps:
            reward = -10  # 踩中陷阱获得负奖励
            done = True
        # 检查是否超过最大步数
        elif self.steps >= self.max_steps:
            done = True
        
        info = {
            'position': self.current_pos,
            'steps': self.steps
        }
        
        return self._get_state(), reward, done, info
    
    def render(self):
        """
        可视化当前环境状态（在控制台打印）
        """
        print("\n" + "-" * (self.size * 2 + 1))
        for i in range(self.size):
            row_str = "|"
            for j in range(self.size):
                pos = (i, j)
                if pos == self.current_pos:
                    row_str += "A|"  # A 代表 Agent（智能体）
                elif pos == self.goal_pos:
                    row_str += "G|"  # G 代表 Goal（目标）
                elif pos in self.traps:
                    row_str += "X|"  # X 代表陷阱
                else:
                    row_str += " |"  # 空格代表空地
            print(row_str)
            print("-" * (self.size * 2 + 1))
        print()
    
    def action_to_str(self, action: int) -> str:
        """将动作编号转换为文字描述"""
        actions = ["上", "下", "左", "右"]
        return actions[action]


# 测试环境
if __name__ == "__main__":
    env = GridWorldEnv(size=4)
    print("=== 环境测试 ===")
    print(f"状态空间大小: {env.observation_space}")
    print(f"动作空间大小: {env.action_space}")
    print(f"起点: {env.start_pos}, 终点: {env.goal_pos}")
    print(f"陷阱: {env.traps}")
    
    # 重置环境
    state = env.reset()
    print(f"\n初始状态: {state}")
    env.render()
    
    # 随机执行几个动作
    for _ in range(5):
        action = np.random.randint(0, env.action_space)
        state, reward, done, info = env.step(action)
        print(f"动作: {env.action_to_str(action)}, 新状态: {state}, 奖励: {reward}, 是否结束: {done}")
        env.render()
        if done:
            break
