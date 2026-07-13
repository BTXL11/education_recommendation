# 教育推荐系统 (Education Recommendation System)

教育推荐系统相关的研究学习项目，包含论文复现代码、学习笔记、参考论文及数据集。

## 项目结构

```
education_recommendation/
├── code/                          # 代码实现
│   ├── DGCPL/                     # IJCAI 2025 - 双图蒸馏概念先修关系学习
│   ├── PREREQ-IAAI-19/            # AAAI IAAI-19 - 从在线教育资源推断概念先修关系
│   ├── Exercise-Recommendation-System-master/  # EDM 2019 - 概念感知知识追踪+RL练习推荐
│   │   ├── DKVMN-CA/              #   知识追踪模型（TensorFlow）
│   │   ├── Exercise Recommendation/  #   强化学习推荐系统（rllab + TRPO）
│   │   ├── paper/EDM2019/         #   论文 LaTeX 源码 + PDF
│   │   ├── PROJECT_EXPLANATION.md #   项目全面详解（小白友好）
│   │   └── RUNNING_GUIDE.md       #   运行指南
│   ├── DecisionTree.ipynb         # 决策树学习（鸢尾花分类）
│   ├── taidibei.ipynb             # 泰迪杯教育平台用户行为数据分析
│   ├── softMax.ipynb              # Softmax 函数学习
│   ├── Source.gv                  # 决策树可视化源文件
│   └── iris_decision_tree.pdf     # 决策树可视化结果
├── data/                          # 数据
│   └── taidibei/                  # 泰迪杯教育平台线上课程用户行为数据集
├── notes/                         # 学习笔记
│   ├── collaborativeFilter.md     # 协同过滤算法详解
│   └── extra.md                   # 决策树 & 帕累托优势
├── thesis/                        # 参考论文（约20篇）
├── best_model/                    # 训练好的最佳模型
├── .gitignore                     # Git 忽略规则（排除大文件夹）
└── README.md
```

## 主要内容

### 1. 推荐算法学习

- **协同过滤算法** — 包括基于用户、基于物品、基于模型的协同过滤，以及热门度修正因子、用户特征权重置信度等改进方法
- **决策树** — 信息增益、基尼指数、MSE 等分裂标准的学习与实践
- **图神经网络** — 图卷积网络（GCN）在概念先修关系预测中的应用

### 2. 论文代码复现

| 论文 | 会议 | 方法 | 代码目录 |
|------|------|------|----------|
| DGCPL: Dual Graph Distillation for Concept Prerequisite Relation Learning | IJCAI 2025 | 双图蒸馏 + 超图神经网络 + 有向GCN | [DGCPL](code/DGCPL/) |
| Inferring Concept Prerequisite Relations from Online Educational Resources | AAAI IAAI-19 | Pairwise Link LDA + 孪生网络 | [PREREQ-IAAI-19](code/PREREQ-IAAI-19/) |
| Concept-Aware Deep Knowledge Tracing and Exercise Recommendation in an Online Learning System | EDM 2019 | DKVMN-CA 知识追踪 + TRPO 强化学习推荐 | [Exercise-Recommendation-System](code/Exercise-Recommendation-System-master/) |

### 3. Exercise-Recommendation-System 详解

EDM 2019 论文的官方代码，包含两大模块：

- **DKVMN-CA** — 概念感知的深度知识追踪模型。基于 DKVMN（动态键值记忆网络），设计了显式对应课程知识概念的记忆结构，利用练习题的知识概念标注信息，追踪学生对每个知识点的掌握状态，预测其答对某道题的概率。
- **Exercise Recommendation** — 基于深度强化学习（TRPO）的个性化练习推荐策略。将训练好的 DKVMN-CA 作为"学生模拟器"，在虚拟环境中训练推荐策略，使其能根据学生的做题历史，推荐最能提高整体知识水平的练习题。

📖 详见项目文档：
- [PROJECT_EXPLANATION.md](code/Exercise-Recommendation-System-master/PROJECT_EXPLANATION.md) — 小白友好的全面详解（原理、架构、公式、代码解读、FAQ）
- [RUNNING_GUIDE.md](code/Exercise-Recommendation-System-master/RUNNING_GUIDE.md) — 运行指南（环境配置、命令行参数、数据格式、完整流程）

### 4. 参考论文

涵盖以下方向：
- 个性化教育资源推荐综述
- 基于协同过滤的课程资源推荐改进
- 知识图谱 + 图卷积神经网络的课程推荐
- 异构信息网络中的非对称推荐
- 基于 LLM 的多智能体学习路径规划
- 个性化课程序列推荐

### 5. 数据集

[泰迪杯](https://www.heywhale.com/mw/dataset/607cfa06f15a1d00171505e3) 教育平台线上课程用户行为数据集，包含：
- 用户注册及登录信息
- 课程选择与学习进度
- 用户地区分布

## 环境依赖

不同子项目需要不同的 Python 环境：

| 项目 | Python | 核心依赖 |
|------|--------|----------|
| DGCPL | 3.10+ | PyTorch 2.1+, PyTorch Geometric 2.5+ |
| PREREQ-IAAI-19 | 3.8+ | PyTorch, scikit-learn |
| Exercise-Recommendation | 3.9 | TensorFlow 2.12 (v1 compat), NumPy 1.23, Gym, Theano, Lasagne |

具体依赖见各子目录的说明文档。

## 快速开始

```bash
# DGCPL 模型训练
cd code/DGCPL
pip install -r environment.txt
bash train.sh

# Exercise-Recommendation-System 运行
# 详见 code/Exercise-Recommendation-System-master/RUNNING_GUIDE.md
cd code/Exercise-Recommendation-System-master
/e/ers_venv/Scripts/python.exe DKVMN-CA/main.py --num_epochs 5 --init_from f

# 运行数据分析
jupyter notebook code/taidibei.ipynb
```

## Git 说明

代码文件夹（`code/DGCPL/`、`code/PREREQ-IAAI-19/`、`code/Exercise-Recommendation-System-master/`）包含大量模型权重、日志、数据文件，已通过 `.gitignore` 排除。如需查看完整代码，请从原始仓库克隆。
