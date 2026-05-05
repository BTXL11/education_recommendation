# 教育推荐系统 (Education Recommendation System)

教育推荐系统相关的研究学习项目，包含论文复现代码、学习笔记、参考论文及数据集。

## 项目结构

```
education_recommendation/
├── code/                          # 代码实现
│   ├── DGCPL/                     # IJCAI 2025 - 双图蒸馏概念先修关系学习
│   ├── PREREQ-IAAI-19/            # AAAI IAAI-19 - 从在线教育资源推断概念先修关系
│   ├── DecisionTree.ipynb         # 决策树学习（鸢尾花分类）
│   ├── taidibei.ipynb             # 泰迪杯教育平台用户行为数据分析
│   ├── Source.gv                  # 决策树可视化源文件
│   └── iris_decision_tree.pdf     # 决策树可视化结果
├── data/                          # 数据
│   └── taidibei/                  # 泰迪杯教育平台线上课程用户行为数据集
├── notes/                         # 学习笔记
│   ├── collaborativeFilter.md     # 协同过滤算法详解
│   └── extra.md                   # 决策树 & 帕累托优势
├── thesis/                        # 参考论文（约20篇）
├── best_model/                    # 训练好的最佳模型
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

### 3. 参考论文

涵盖以下方向：
- 个性化教育资源推荐综述
- 基于协同过滤的课程资源推荐改进
- 知识图谱 + 图卷积神经网络的课程推荐
- 异构信息网络中的非对称推荐
- 基于 LLM 的多智能体学习路径规划
- 个性化课程序列推荐

### 4. 数据集

[泰迪杯](https://www.heywhale.com/mw/dataset/607cfa06f15a1d00171505e3) 教育平台线上课程用户行为数据集，包含：
- 用户注册及登录信息
- 课程选择与学习进度
- 用户地区分布

## 环境依赖

部分代码需要以下 Python 环境：

- Python 3.10+
- PyTorch 2.1+
- PyTorch Geometric 2.5+
- scikit-learn 1.2+
- pandas, numpy

具体依赖见各子目录的 `environment.yml` 文件。

## 快速开始

```bash
# 安装依赖
pip install -r code/DGCPL/environment.txt

# 训练 DGCPL 模型
cd code/DGCPL
bash train.sh

# 运行数据分析
jupyter notebook code/taidibei.ipynb
```
