---
name: project_data_leakage
description: 监督模型存在"弱标签泄露" — RNA/ATAC cluster (训练) 与 Joint cluster (评估) 同源
metadata:
  type: project
---
## 核心结论
监督模型的高 NMI **存在弱泄露** (label leakage)，因为训练用的 RNA_clusters / ATAC_clusters 和评估目标 Joint_clusters 来自同一份 meta，互相相关。
**Why**: 用户敏锐地察觉到这一点 — "我感觉监督方法似乎有数据泄露"，这是对的。
**How to apply**: 报告结果时必须同时给出无监督基线和"作弊基线"作为上下界；不要把监督 NMI=0.53 当成纯几何质量的体现。

## 量化分析 (meta.csv 标签相关性)

| | RNA_clusters vs Joint | ATAC_clusters vs Joint |
|---|---|---|
| E13 | NMI=0.636 / ARI=0.504 | NMI=0.366 / ARI=0.232 |
| P21 | NMI=0.661 / ARI=0.558 | NMI=0.527 / ARI=0.481 |

## 反证: 模型并非简单复制 RNA 标签
- "作弊基线" (RNA_clusters 直接当预测): avg NMI ≈ 0.649
- 监督模型实际: avg NMI = 0.529
- 监督版**比作弊基线低 ~0.12** → recon/align/spatial/gp 几项自监督损失把它拽偏，没最大化利用 cluster head 信号

## 真实基线判断
- **无监督 NMI=0.42 才是干净的天花板** (不见任何标签时学到的几何质量)
- 监督 0.53 里至少有 0.1+ 来自这种弱泄露相关性

## 消除泄露感的方向
- 评估用与训练 label 不同源的下游任务: marker gene 重建、GP 边召回、held-out spot 表达预测
- 训练改对比/掩码自监督 (走无监督路线，见 [[project_unsupervised_improvements]])
- 报告时同时列 RNA→Joint / ATAC→Joint 作为上下界参照
- **走半监督路线**: 见 [[project_semi_supervised]], 10% 标签下 NMI=0.48 仍显著高于 unsup, 泄露强度按比例稀释
