---
name: project_graph_building
description: 空间多组学异构图建图+模型训练，有监督/supervised+无监督/unsupervised双版本，支持GPU(5070Ti)
type: project
originSessionId: 53a1124e-f5f2-4899-8128-60eb00a471fc
---
## 项目结构
```
scg/
├── common/loader.py           # 共享数据加载 (SVD+GPU+预计算边值)
├── supervised/                # 有监督版 (含RNA/ATAC cluster heads)
│   ├── model.py               #   SpatialHeteroConv + SupervisedEncoder + ClusterHeads
│   └── train.py               #   训练入口 (--device cuda --amp)
├── unsupervised/              # 无监督版 (纯自监督)
│   ├── model.py               #   SpatialHeteroConv + UnsupervisedEncoder
│   └── train.py               #   训练入口
├── build_graph.py             # 建图脚本 (共用)
├── graph_combined/            # 图数据 (共用)
├── output_sup/                # 监督版输出
├── output_unsup/              # 无监督版输出
└── marsgt/                    # MarsGT参考代码 (独立git仓库)
```

### 两个版本区别
| | supervised | unsupervised |
|------|:--:|:--:|
| RNA/ATAC cluster预测 | ✅ 参与训练 | ❌ |
| 训练信号 | 自监督 + cluster labels | 纯自监督 (4 losses) |
| ClusterHeads | ✅ 6个 (3时间点×2模态) | ❌ |
| Joint_clusters | 仅评估 (held out) | 仅评估 |

### 数据集
- E13: 2187 spots, P21: 2373 spots, P22: 9215 spots
- 统一图: 13775 spots + 10192 genes + 16864 peaks = 40831 nodes
- 4种边: SS(40K) + SG(1.38M) + SP(5.51M) + GP(10.6K)
- Spot时间标签: 0=E13, 1=P21, 2=P22
- Spot-Spot KNN 仅在各自时间点内部 (k=5), **不跨时间连边**
- Gene/Peak名共享

### 节点ID布局
- Spot: [0, 13775); Gene: [13775, 23967); Peak: [23967, 40831)

### 损失函数
**无监督版 (4项)**:
L = L_recon + λ1·L_align + λ2·L_spatial + λ3·L_gp

**监督版 (6项)**:
L = L_recon + λ1·L_align + λ2·L_spatial + λ3·L_gp + λ4·L_rna + λ5·L_atac

| 损失 | 权重 | 含义 |
|------|------|------|
| L_recon | 1.0 | embedding内积重建表达值 (采样30K边) |
| L_align | 0.3 | RNA-view vs ATAC-view InfoNCE跨模态对齐 |
| L_spatial | 0.2 | KNN近邻嵌入距离加权平滑 |
| L_gp | 1.0 | Gene-Peak连边BCE (5:1负采样) |
| L_rna | 0.5 | 每时间点独立预测RNA_clusters (监督) |
| L_atac | 0.5 | 每时间点独立预测ATAC_clusters (监督) |

### 模型架构 (GPU-ready)
- **FourierPositionEncoding**: 坐标(x,y) → 64维傅里叶特征
- **SpatialHeteroConv**: 多头注意力 + 距离衰减核 (Spot-Spot)
  - Spot-Spot: distance-aware multi-head attention (可学习σ)
  - 其他边: 标准multi-head attention (num_heads=4, d_k=64)
  - Residual + LayerNorm + GELU FFN per node type
- 2层卷积, hidden_dim=256
- SVD降维 (TruncatedSVD dim=100~200, n_iter=5)
  - RNA方差解释率~9% (50d)→需增大dim
  - ATAC方差解释率~95%
- 支持混合精度 (--amp)

### 关键设计决策
1. **跨时间Spot-Spot不连边**: 通过共享gene/peak节点间接传递信息
2. **Joint_clusters不参与训练**: 纯当外部评估指标 (E13+P21共4560 spots)
3. **Cluster标签按时间点独立**: 不同时间点标签无对应关系 (E13 ATAC用A前缀, P21/P22用C前缀)
4. **边值预计算**: loader.py在加载时预计算SG/SP边上的表达值，训练时直接采样避免sparse lookup
5. **P22无Joint_clusters**: 仅E13+P21参与NMI/ARI评估

### v1 测试结果 (10 epochs, SVD dim=50, CPU, supervised)
- NMI=0.419, ARI=0.248
- 架构方向验证通过，主要瓶颈是SVD维度太低和epoch太少

### v2 完整结果 (250 epochs, hidden=128, GPU)
- Supervised best: avg NMI=0.5423 (per_time KMeans)
- Unsupervised best: avg NMI=0.4192 (joint_e13p21 KMeans) — **真实基线** (无泄露)
- 详细 6 指标 + Leiden 对比见 [[project_experiments_results]]

### GPU使用 (5070 Ti, 16GB)
**hidden=256 会 OOM**, 使用 hidden=128:
```bash
python unsupervised/train.py --device cuda --amp --epochs 250 --hidden 128 --svd_dim 100
python supervised/train.py --device cuda --amp --epochs 250 --hidden 128 --svd_dim 100
```

### 评估指标 (6 个)
NMI / AMI / ARI / FMI / MI / ACC，ACC 通过 Hungarian matching (scipy.linear_sum_assignment) 计算。
实现位置: `supervised/model.py` + `unsupervised/model.py` 里 `_cluster_acc()` 和 `_all_metrics()`。

### 聚类口径 (3 种 cluster_mode)
- **per_time**: E13/P21 各自独立 KMeans (历史最稳, sup NMI=0.5423)
- **joint_e13p21**: E13+P21 池化后一起 KMeans, P22 不参与
- **joint_all**: E13+P21+P22 全部池化后 KMeans (P22 干扰 unsup 最敏感)
训练参数: `--joint_cluster_e13p21` 或 `--joint_cluster_all`

### Leiden 聚类 (生信常用)
`eval_leiden.py` — 加载已训好的 spot_emb.npy 跑 Leiden 后处理 (scanpy + leidenalg)，不需要重训。
HGNA conda env 有 scanpy 1.11 + leidenalg 0.11。
扫 resolution=0.5/1.0/1.5。详见 [[project_experiments_results]]。

### 可视化
`visualize.py` 出 3 张图: umap_by_time / umap_by_cluster / spatial_clusters。详见 [[project_visualization_findings]]。

### 实验结果 / 待改进
- 监督 vs 无监督完整对比: [[project_experiments_results]]
- 监督模型有弱标签泄露: [[project_data_leakage]]
- 无监督改进方向: [[project_unsupervised_improvements]]
