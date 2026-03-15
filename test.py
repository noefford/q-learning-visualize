"""
测试脚本：测试训练好的 Q-Learning 智能体
"""
import numpy as np
from grid_world_env import GridWorldEnv
from q_learning_agent import QLearningAgent


def test_agent(
    model_path: str = "q_learning_model.pkl",
    n_episodes: int = 5,
    render: bool = True,
    delay: float = 0.5
):
    """
    测试训练好的智能体
    
    Args:
        model_path: 模型文件路径
        n_episodes: 测试轮数
        render: 是否可视化
        delay: 每步之间的延迟（秒）
    """
    # 创建环境
    env = GridWorldEnv(size=4)
    
    # 创建智能体并加载模型
    agent = QLearningAgent(
        n_states=env.observation_space,
        n_actions=env.action_space
    )
    agent.load(model_path)
    
    # 测试时使用贪婪策略（关闭探索）
    print("\n" + "=" * 60)
    print("开始测试智能体")
    print("=" * 60)
    
    total_rewards = []
    total_steps = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\n--- 测试轮次 {episode + 1}/{n_episodes} ---")
        
        if render:
            print(f"初始状态:")
            env.render()
            import time
            time.sleep(delay)
        
        while not done:
            # 测试时使用贪婪策略（不探索）
            action = agent.choose_action(state, training=False)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            if render:
                print(f"动作: {env.action_to_str(action)}, 奖励: {reward}")
                env.render()
                time.sleep(delay)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if episode_steps > 100:  # 防止无限循环
                print("达到最大步数限制")
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        
        # 判断结果
        final_pos = info['position']
        if final_pos == env.goal_pos:
            result = "✓ 成功到达终点！"
        elif final_pos in env.traps:
            result = "✗ 踩中陷阱！"
        else:
            result = "○ 步数耗尽"
        
        print(f"结果: {result} | 总奖励: {episode_reward} | 步数: {episode_steps}")
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"测试轮数: {n_episodes}")
    print(f"平均奖励: {np.mean(total_rewards):.2f}")
    print(f"平均步数: {np.mean(total_steps):.2f}")
    print(f"成功次数: {sum(1 for r in total_rewards if r > 0)}/{n_episodes}")
    print("=" * 60)


def print_q_table(agent: QLearningAgent, env: GridWorldEnv):
    """
    打印Q表，展示智能体学到的策略
    """
    print("\n" + "=" * 60)
    print("Q表展示 (每个状态下各动作的Q值)")
    print("=" * 60)
    
    q_table = agent.get_q_table()
    actions = ["上", "下", "左", "右"]
    
    print("\n状态 |  上   |  下   |  左   |  右   | 最佳动作")
    print("-" * 55)
    
    for state in range(env.observation_space):
        q_values = q_table[state]
        best_action = np.argmax(q_values)
        
        # 获取坐标
        row = state // env.size
        col = state % env.size
        
        # 特殊位置标记
        pos_marker = ""
        if (row, col) == env.goal_pos:
            pos_marker = " [G]"
        elif (row, col) in env.traps:
            pos_marker = " [X]"
        elif (row, col) == env.start_pos:
            pos_marker = " [S]"
        
        print(f"({row},{col}){pos_marker:4} | {q_values[0]:6.2f} | {q_values[1]:6.2f} | "
              f"{q_values[2]:6.2f} | {q_values[3]:6.2f} | {actions[best_action]}")
    
    print("\n说明:")
    print("  [S] = 起点 (Start)")
    print("  [G] = 目标 (Goal)")
    print("  [X] = 陷阱 (Trap)")
    print("=" * 60)


def visualize_policy(env: GridWorldEnv, agent: QLearningAgent):
    """
    可视化智能体的策略（在每个位置显示最佳动作）
    """
    print("\n" + "=" * 60)
    print("策略可视化 (每个位置的最佳动作)")
    print("=" * 60)
    
    action_arrows = {
        0: "↑",  # 上
        1: "↓",  # 下
        2: "←",  # 左
        3: "→"   # 右
    }
    
    print("\n" + "-" * (env.size * 4 + 1))
    for i in range(env.size):
        row_str = "|"
        for j in range(env.size):
            state = i * env.size + j
            pos = (i, j)
            
            if pos == env.goal_pos:
                row_str += " G |"  # 目标
            elif pos in env.traps:
                row_str += " X |"  # 陷阱
            else:
                # 获取最佳动作
                best_action = np.argmax(agent.q_table[state])
                row_str += f" {action_arrows[best_action]} |"
        print(row_str)
        print("-" * (env.size * 4 + 1))
    
    print("\n图例: ↑上 ↓下 ←左 →右  G目标  X陷阱")
    print("=" * 60)


def manual_play(env: GridWorldEnv):
    """
    手动玩游戏（让用户自己尝试）
    """
    print("\n" + "=" * 60)
    print("手动模式：你可以自己控制智能体")
    print("=" * 60)
    print("控制方式: w=上, s=下, a=左, d=右, q=退出")
    print("=" * 60)
    
    state = env.reset()
    env.render()
    
    total_reward = 0
    steps = 0
    
    while True:
        key = input("\n输入动作 (w/a/s/d): ").lower().strip()
        
        if key == 'q':
            print("退出游戏")
            break
        
        # 映射按键到动作
        action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}
        
        if key not in action_map:
            print("无效输入！请使用 w/a/s/d")
            continue
        
        action = action_map[key]
        state, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        print(f"动作: {env.action_to_str(action)}, 奖励: {reward}, 累计奖励: {total_reward}")
        env.render()
        
        if done:
            final_pos = info['position']
            if final_pos == env.goal_pos:
                print(f"\n✓ 恭喜！你成功到达终点！总奖励: {total_reward}, 步数: {steps}")
            elif final_pos in env.traps:
                print(f"\n✗ 踩中陷阱！游戏结束。总奖励: {total_reward}")
            else:
                print(f"\n○ 步数耗尽！总奖励: {total_reward}")
            
            play_again = input("\n再玩一次? (y/n): ").lower()
            if play_again == 'y':
                state = env.reset()
                total_reward = 0
                steps = 0
                env.render()
            else:
                break


if __name__ == "__main__":
    import sys
    
    # 如果没有参数，执行完整测试
    if len(sys.argv) == 1 or sys.argv[1] == "test":
        # 测试智能体
        test_agent(
            model_path="q_learning_model.pkl",
            n_episodes=5,
            render=True,
            delay=0.5
        )
    
    # 创建环境和智能体用于展示
    env = GridWorldEnv(size=4)
    agent = QLearningAgent(
        n_states=env.observation_space,
        n_actions=env.action_space
    )
    
    try:
        agent.load("q_learning_model.pkl")
        
        # 打印Q表
        print_q_table(agent, env)
        
        # 可视化策略
        visualize_policy(env, agent)
        
    except FileNotFoundError:
        print("\n模型文件不存在，请先运行 train.py 进行训练！")
    
    # 询问是否手动玩
    print("\n" + "=" * 60)
    choice = input("是否想要手动试玩? (y/n): ").lower()
    if choice == 'y':
        manual_play(env)
