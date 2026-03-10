# Gitcoin GG24 Deep Funding Contest - Level III Model

celestial 提交仓库

## 模型概述
- 算法：Reverse Personalized PageRank (alpha=0.85)
- 数据：pairs_to_predict.csv (3677 deps, 83 unique repo), unweighted_graph.csv (31 seed graph)
- 处理：反向图传播 + 强制 sum=1 + 最小权重平滑 1e-5
- 提交文件：submission_dependency_repo_weight.csv

## 文件说明
- level3_strong_baseline_final.py：核心代码
- submission_dependency_repo_weight.csv：最终提交 CSV
- 说明Level III 模型.docx：详细 writeup

## 运行方式
1. 安装依赖：pip install pandas numpy networkx
2. 运行：python level3_strong_baseline_final.py
3. 输出：submission_dependency_repo_weight.csv

欢迎交流 💕
