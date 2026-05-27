---
name: project_unsupervised_improvements
description: 提升无监督模型效果的待办方向 (按风险/收益排序)
metadata:
  type: project
---
## 背景
监督模型有弱标签泄露 (见 [[project_data_leakage]])，**无监督 NMI=0.42 才是真实基线**。下面方向按"零代码改动 → 较多改动"排序。

## A. 提高 SVD 输入维度 (零代码，预期收益最大)
**Why**: 当前 `svd_dim=100` 仅解释 ~13% RNA 方差，**扔掉了 87% 的信号**。memory 早就标注过这一点。
**How to apply**:
```bash
python unsupervised/train.py --epochs 250 --hidden 128 --svd_dim 300 \
    --joint_cluster_e13p21 --device cuda --output_dir output_unsup_svd300
```
代价: 显存稍涨。预期 NMI +0.02~0.05。

## B. 增大 spatial loss 权重 (零代码)
**Why**: epoch 250 时 recon=0.16、align=0.35、gp=0.08 都收敛了，但 **spatial 还停在 2.44** — `lambda_spatial=0.2` 太弱，spatial 几何根本没拟合下去。E13 spot 稀疏，最依赖 spatial 平滑。
**How to apply**:
```bash
python unsupervised/train.py --epochs 250 --lambda_spatial 1.0 ...
```
预期对 E13 帮助更大 (现在 E13 NMI=0.39 落后 P21 的 0.45)。

## C. MAE-style 掩码重建 (中等代码量，预期收益最大)
**Why**: 当前 `L_recon` 用 embedding 内积重建**已有连边**的表达 — 这是被动重建。改成主动屏蔽 15% gene/peak 节点，强制 encoder 从邻居恢复 — graph SSL 标配 (GraphMAE / S2GAE)。
**How to apply**: 比对比学习实现更简单，约 40 行代码。

## D. 图对比学习 (中等代码量，可能最大收益)
**Why**: 现在 align 只是 RNA-view vs ATAC-view 跨模态对齐。加 **spot 级 dual-view contrastive**: 同一 spot 两次图增强 (drop edge / mask feature)，让两个 view 的 embedding 拉近、和其他 spot 推远。无监督图表征 SOTA 路线 (GRACE / BGRL / DGI)。
**How to apply**: 约 60 行代码。

## E. 处理 P22 主导问题
**Why**: P22 占 67% spot (9215/13775)，loss 求和被它支配，E13/P21 信号被稀释 (见 [[project_visualization_findings]] 中 unsup UMAP P22 统治视野中央)。
**How to apply**: 采样平衡 — 每 batch / 每 epoch 对 P22 做 spot 下采样到 E13/P21 数量级；或 loss 按时间点 reweight。
