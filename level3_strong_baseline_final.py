# level3_strong_baseline_final.py
# 最终真实完整版：基于 pairs_to_predict.csv 的 83 repo + 3677 deps

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

print("=== Deep Funding Level III - 真实完整版 (83 repo + 3677 deps) ===")

folder = Path.cwd()

# 加载 pairs_to_predict.csv（完整 3677 行，83 unique repo）
df_pairs = pd.read_csv(folder / "pairs_to_predict.csv")
df_pairs = df_pairs.rename(columns={'dependency': 'source', 'repo': 'target'})
df_pairs['source'] = df_pairs['source'].astype(str).str.strip()
df_pairs['target'] = df_pairs['target'].astype(str).str.strip()

print(f"pairs_to_predict.csv 总行数: {len(df_pairs)}")
print(f"unique target (repo): {df_pairs['target'].nunique()}")

# 所有 target 来自 pairs（83 个）
all_targets = sorted(df_pairs['target'].unique())

# 加载 unweighted_graph.csv 并拼接
df_graph_raw = pd.read_csv(folder / "unweighted_graph.csv")
df_graph = df_graph_raw.copy()
df_graph['target'] = df_graph['seed_repo_owner'].astype(str) + '/' + df_graph['seed_repo_name'].astype(str)
df_graph['source'] = df_graph['package_repo_owner'].astype(str) + '/' + df_graph['package_repo_name'].astype(str)
df_graph['target'] = df_graph['target'].str.strip()
df_graph['source'] = df_graph['source'].str.strip()

df_graph = df_graph.dropna(subset=['target', 'source'])
df_graph = df_graph[(df_graph['target'].str.len() > 0) & (df_graph['source'].str.len() > 0)]

# 构建反向图
print("构建反向图...")
G = nx.DiGraph()
for _, row in df_graph.iterrows():
    G.add_edge(row['target'], row['source'])

submission = []

for i, target in enumerate(all_targets, 1):
    if i % 10 == 0 or i == len(all_targets):
        print(f"处理进度: {i}/{len(all_targets)}")

    deps = df_pairs[df_pairs['target'] == target]['source'].unique().tolist()

    if not deps:
        print(f"{target} 无依赖 → 跳过（但 pairs 中不应出现）")
        continue

    if target not in G.nodes():
        w = 1.0 / len(deps)
        for dep in deps:
            submission.append([dep, target, round(w, 6)])
        continue

    try:
        pr = nx.pagerank(G, alpha=0.85, personalization={target: 1.0},
                         max_iter=500, tol=1e-08)
    except Exception as e:
        print(f"PageRank 失败 ({target}) → 用均匀")
        w = 1.0 / len(deps)
        for dep in deps:
            submission.append([dep, target, round(w, 6)])
        continue

    scores = {dep: max(pr.get(dep, 0), 1e-6) for dep in deps}
    total = sum(scores.values()) or 1.0
    for dep in deps:
        weight = scores[dep] / total
        submission.append([dep, target, round(weight, 6)])

df_submit = pd.DataFrame(submission, columns=['source', 'target', 'weight'])

# 最终归一化 + 平滑
if not df_submit.empty:
    df_submit['weight'] = df_submit.groupby('target')['weight'].transform(lambda x: x / x.sum())
    df_submit['weight'] = df_submit.groupby('target')['weight'].transform(lambda x: np.maximum(x, 1e-5))

# 输出列头 dependency,repo,weight（最兼容）
df_submit = df_submit.rename(columns={'source': 'dependency', 'target': 'repo'})

df_submit.to_csv('submission_dependency_repo_weight.csv', index=False)
print(f"\n生成 submission_dependency_repo_weight.csv （{len(df_submit)} 行）")
print(f"覆盖 repo 数量: {df_submit['repo'].nunique()} (应=83)")

print("\n前15行预览：")
print(df_submit.head(15))

print("\n完成！")
print("现在用 submission_dependency_repo_weight.csv 上传平台。")
print("列头 dependency,repo,weight，无前缀，匹配原始 pairs。")
print("如果还 missing pairs：说明 pairs_to_predict.csv 还是不全，重新下载完整版。")
print("如果成功，立刻写 PDF writeup 提交！")