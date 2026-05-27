---
name: project_experiments_results
description: 监督/无监督训练实验结果对比 (250 epoch, hidden=128) + Leiden vs KMeans
metadata:
  type: project
---
## 实验结果总表 (E13+P21 评估, 6 指标)

### 监督模型 (output_sup_joint/, 250 epoch, hidden=128)

| 聚类口径 | 算法 | n_clusters | avg NMI | avg ARI |
|---|---|---|---|---|
| per_time | KMeans (昨天 300ep) | 9/9 | **0.5423** | **0.4427** |
| joint_all | KMeans (今天 250ep) | 18 | 0.5286 | 0.4372 |
| per_time | Leiden res=1.0 | 14/12 | 0.5329 | 0.4355 |
| joint_e13p21 | Leiden res=1.5 | 25 | 0.5338 | 0.4236 |
| joint_all | Leiden res=1.0 | 21 | 0.5050 | 0.4003 |

E13 best: NMI=0.5035 / ARI=0.3726; P21 best: NMI=0.5537 / ARI=0.5017 (KMeans joint_all)

### 无监督模型 (output_unsup_joint_e13p21/, 250 epoch, hidden=128)

| 聚类口径 | 算法 | n_clusters | avg NMI | avg ARI |
|---|---|---|---|---|
| joint_e13p21 | KMeans | 18 | **0.4192** | 0.3140 |
| joint_all | KMeans | 18 | 0.392 | 0.291 |
| per_time | Leiden res=1.5 | 11/14 | 0.3794 | **0.3278** |
| joint_e13p21 | Leiden res=1.0 | 10 | 0.3668 | 0.2506 |
| joint_all | Leiden res=0.5 | 4 | 0.004 | -0.001 (崩) |

## 关键观察
- **监督嵌入上 Leiden ≈ KMeans** (NMI 差 < 0.01)，几何已足够清晰
- **无监督嵌入上 Leiden < KMeans** (NMI 0.42→0.38)，簇分离不够紧密
- **per_time 是最稳的口径**，joint_all 在 unsup 上对 P22 干扰最敏感
- **Leiden resolution 敏感**: unsup+joint_all res=0.5 几乎崩溃 (4 簇全归一类), res=1.0–1.5 是 sweet spot
- 监督 vs 无监督差距集中在: **P21/P22 区分度** 和 **P22 内部异质性** (见 [[project_visualization_findings]])

## 训练历史 (监督 joint_all 250 epoch)
- Epoch 10: NMI=0.353, ARI=0.253 (起步稍慢, cluster head 预热)
- Epoch 50: NMI=0.359, ARI=0.257
- Epoch ~70: NMI=0.434, ARI=0.309 (开始拉开无监督)
- Epoch 100: NMI=0.442, ARI=0.317 (显著超过 unsup 同期 0.382)
- Epoch 250: NMI=0.5286, ARI=0.4372 (best)

## 评估脚本
- `eval_leiden.py` — 加载 spot_emb.npy 跑 Leiden, 三种 cluster_mode × 多个 resolution
- 评估日志: `eval_leiden.log`
- 评估时 RNA/ATAC/Joint clusters 关系见 [[project_data_leakage]]
