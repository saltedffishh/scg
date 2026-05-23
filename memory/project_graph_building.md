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

### GPU使用 (5070 Ti, 16GB)
```bash
python unsupervised/train.py --device cuda --amp --epochs 300 --svd_dim 200
python supervised/train.py --device cuda --amp --epochs 300 --svd_dim 200
```

### 待改进
- [ ] GPU完整训练并对比监督vs无监督
- [ ] SVD dim增大 (100→200+) 提升RNA方差解释率
- [ ] 消融实验验证每个loss贡献
- [ ] 跨时间spot对应方法
- [ ] Gene-Peak score阈值确定
