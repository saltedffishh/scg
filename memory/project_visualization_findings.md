---
name: project_visualization_findings
description: UMAP + spatial 可视化产物及视觉发现 (监督 vs 无监督)
metadata:
  type: project
---
## 可视化脚本
`visualize.py` — 加载 spot_emb.npy + leiden_clusters.npy，输出 3 张图:
- `umap_by_time.png` — UMAP 按时间点着色 (E13/P21/P22)
- `umap_by_cluster.png` — UMAP 按 Leiden cluster 着色
- `spatial_clusters.png` — 三个时间点空间坐标分群

产物路径:
```
output_sup_joint/figs/             # 监督 (Leiden res=1.0, 21 clusters)
output_unsup_joint_e13p21/figs/    # 无监督 (Leiden res=1.0, 8 clusters)
```

## 视觉对比

| | supervised | unsupervised |
|---|---|---|
| Leiden cluster 数 (res=1.0) | **21** | **8** |
| E13 vs (P21,P22) UMAP 分离 | 完全分开 | 完全分开 (更小一坨) |
| P21 vs P22 UMAP 分离 | 有可见亚结构 | 几乎完全融合 |
| 空间分群锐利度 | 同色形成清晰局部区域 | 同色大片铺，亚结构弱 |

## 关键观察
1. **同样 res=1.0, unsup 只切 8 簇, 监督切 21 簇** — 监督 embedding 密度峰更多更紧致；无监督嵌入更"平"，没那么多细密度峰。
2. **E13 完全独立成簇** — 这是数据本身的强信号 (胚胎期 vs 成体期表达差异巨大)，自监督也能从 RNA/ATAC 共表达模式中学到。
3. **P22 在 unsup UMAP 里统治视野中央** — P22 占总数 67% (9215/13775)，自监督损失被它主导 (见 [[project_unsupervised_improvements]] 方向 E)。
4. **空间图 spatial loss 在起作用** — 同色 spot 形成局部连续区域，不是随机散布 (L_spatial KNN 平滑奏效)。但 unsup 上 P22 大面积只用一两种颜色 → P22 内部异质性没学出来。

## 两组可视化结论
佐证 NMI 数字 (sup 0.53 vs unsup 0.42)，主要差距在:
- **P21/P22 区分度**
- **P22 内部异质性**
