# 🎮 Q-Learning 可视化教程

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/RL-Q--Learning-orange.svg" alt="RL">
</p>

<p align="center">
  <b>零基础入门强化学习 | 交互式可视化训练 | 完整代码实现</b>
</p>

<p align="center">
  🌐 <a href="#-在线体验">在线体验</a> • 
  📖 <a href="#-文档教程">文档教程</a> • 
  🚀 <a href="#-快速开始">快速开始</a> • 
  📷 <a href="#-项目展示">项目展示</a>
</p>

---

## ✨ 项目特色

### 🎯 适合初学者
- **零基础友好**：从概念到代码，循序渐进
- **中文注释**：所有代码都有详细中文注释
- **交互式学习**：通过可视化直观理解算法

### 🎨 多种可视化方式
- **🌐 网页版**：无需安装，浏览器直接运行
- **📊 桌面版**：Matplotlib实时更新Q表
- **📈 训练曲线**：奖励、步数、探索率变化
- **🧮 公式展示**：实时显示Q值计算过程

### 📚 完整学习资料
- **入门教程**：HTML格式的精美教程文档
- **深度解析**：数学原理和收敛性分析
- **代码示例**：从基础到进阶的完整代码

---

## 🚀 快速开始

### 方式一：网页版（推荐 ⭐）

无需安装任何依赖，直接在浏览器中打开：

```bash
# 双击打开即可
web_visual.html
```

**功能特点**：
- ✅ 实时训练可视化
- ✅ 交互式控制（开始/暂停/单步/批量训练）
- ✅ 响应式设计，支持移动端

---

### 方式二：Python 版本

#### 环境要求
- Python 3.8+
- NumPy
- Matplotlib

#### 安装依赖
```bash
pip install numpy matplotlib
```

#### 运行项目

```bash
# 1. 基础训练（命令行）
python train.py

# 2. 可视化训练（实时展示Q表更新）⭐ 推荐
python train_visual_v2.py

# 3. 测试训练好的智能体
python test_visual.py

# 4. 进阶：DQN算法
python cartpole_dqn.py
```

---

## 📁 项目结构

```
q-learning-visualization/
│
├── 📘 教程文档
│   ├── 教程.html                    # 入门教程（精美排版）
│   └── Q-Learning深度解析.html       # 数学原理深度解析
│
├── 🎮 核心代码
│   ├── grid_world_env.py            # 网格世界环境
│   ├── q_learning_agent.py          # Q-Learning智能体
│   └── train.py                     # 基础训练脚本
│
├── 🎨 可视化工具
│   ├── web_visual.html              # 网页版可视化 ⭐
│   ├── train_visual_v2.py           # 桌面版可视化
│   ├── test_visual.py               # 智能体测试
│   └── visualize_learning.py        # 学习过程可视化
│
├── 🔬 进阶示例
│   └── cartpole_dqn.py              # DQN深度Q网络
│
├── 🛠️ 配置文件
│   ├── README.md                    # 本文件
│   ├── .gitignore                   # Git忽略文件
│   └── requirements.txt             # Python依赖
│
└── 📊 输出文件（自动生成）
    ├── q_learning_model.pkl         # 训练好的模型
    └── training_results.png         # 训练结果图
```

---

## 📖 核心概念

### Q-Learning 算法公式

```
Q(s,a) = Q(s,a) + α × [r + γ × max(Q(s',a')) - Q(s,a)]
```

### 参数说明

| 参数 | 符号 | 说明 | 常用值 |
|------|------|------|--------|
| 学习率 | α | 控制Q值更新幅度 | 0.1 ~ 0.3 |
| 折扣因子 | γ | 未来奖励的重要性 | 0.9 ~ 0.99 |
| 探索率 | ε | 随机探索的概率 | 1.0 → 0.01 |

### 环境设置

```
4×4 网格世界

[S] [ ] [ ] [ ]    S = 起点 (Start)
[ ] [X] [ ] [ ]    X = 陷阱 (Trap)   奖励: -10
[ ] [ ] [X] [ ]    G = 终点 (Goal)   奖励: +10
[ ] [ ] [ ] [G]    每步惩罚: -1

状态数: 16 (4×4)
动作数: 4 (上/下/左/右)
```

---

## 📷 项目展示

### 1️⃣ 网页版可视化界面

<p align="center">
  <b>主界面 - 实时训练监控</b><br>
  网格世界 | Q值表 | 训练曲线 | 控制面板
</p>

**功能亮点**：
- 🎮 **交互式控制**：开始/暂停/单步/批量训练
- 📊 **实时更新**：Q表、策略、统计数据实时刷新
- 🧮 **公式展示**：显示当前更新的计算过程
- 📈 **动态图表**：Plotly绘制的交互式图表

---

### 2️⃣ 桌面版可视化

```bash
python train_visual_v2.py
```

**界面布局**：
- 左上：Q表热力图
- 中上：当前更新详情
- 右上：Q-Learning公式
- 左中：网格世界
- 中中：策略可视化
- 右中：训练统计
- 底部：控制按钮

---

### 3️⃣ 训练结果示例

**奖励曲线**：
```
奖励
  ↑
10├                    ●●●●●
  │              ●●●●●
 0├        ●●●●●
  │   ●●●●
-10├●●
  └────────────────────────→ 回合
```

- **初期**：奖励波动大（探索阶段）
- **中期**：奖励逐渐上升（学习阶段）
- **后期**：奖励稳定在较高水平（收敛）

---

## 🎯 使用示例

### 基础用法

```python
from grid_world_env import GridWorldEnv
from q_learning_agent import QLearningAgent

# 创建环境
env = GridWorldEnv(size=4)

# 创建智能体
agent = QLearningAgent(
    n_states=16,
    n_actions=4,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# 训练
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state, training=True)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    agent.update_epsilon()
    print(f"Episode {episode}: Reward = {total_reward}")
```

---

## 🔧 自定义配置

### 修改训练参数

在 `train.py` 中调整：

```python
agent = QLearningAgent(
    learning_rate=0.1,      # 学习率（太大不稳定，太小学得慢）
    gamma=0.99,             # 折扣因子（重视长期回报）
    epsilon=1.0,            # 初始探索率（1.0=完全随机）
    epsilon_decay=0.995,    # 衰减速度（0.995=慢衰减）
    epsilon_min=0.01        # 最小探索率（保持一定探索）
)
```

### 修改环境

在 `grid_world_env.py` 中：

```python
env = GridWorldEnv(
    size=5,                    # 改为5×5网格
    traps=[(1,1), (2,2), (3,3)] # 添加更多陷阱
)
```

---

## 📚 学习路径

### 阶段一：入门（1-2小时）
1. 📖 阅读 [教程.html](./教程.html) 了解基础概念
2. 🎮 打开 `web_visual.html` 观察训练过程
3. 📝 运行 `train.py` 完成第一次训练

### 阶段二：理解（2-3小时）
1. 🔍 阅读 [Q-Learning深度解析.html](./Q-Learning深度解析.html)
2. 🧮 理解贝尔曼方程和TD学习
3. 🔬 对比 Q-Learning vs SARSA

### 阶段三：实践（3-5小时）
1. 🎨 使用 `train_visual_v2.py` 观察Q表更新
2. ⚙️ 调整参数（学习率、折扣因子等）观察效果
3. 🧪 修改环境（网格大小、陷阱位置）

### 阶段四：进阶（5小时+）
1. 🧠 学习 `cartpole_dqn.py` 中的DQN算法
2. 📝 尝试实现 Double Q-Learning
3. 🚀 应用到其他环境（OpenAI Gym）

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **编程语言** | Python 3.8+ |
| **数值计算** | NumPy |
| **数据可视化** | Matplotlib, Plotly.js |
| **前端框架** | Tailwind CSS |
| **文档格式** | HTML5 |

---

## 🤝 如何贡献

欢迎提交 Issue 和 Pull Request！

### 贡献方式
1. 🐛 报告 Bug
2. 💡 提出新功能建议
3. 📝 改进文档
4. 🔧 优化代码

### 提交规范
- 使用清晰的提交信息
- 确保代码通过基础测试
- 更新相关文档

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

你可以自由地：
- ✅ 使用
- ✅ 修改
- ✅ 分发
- ✅ 商用

只需保留原始许可证和版权声明。

---

## 🙏 致谢

- 感谢 [OpenAI](https://openai.com/) 提供的 Gym 环境
- 感谢 [Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf) 的经典教材
- 感谢所有开源社区的贡献者
