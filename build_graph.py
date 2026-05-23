"""
================================================================================
建图脚本：为空间多组学数据构建包含4种关系的异构图
================================================================================

【背景】
本项目处理小鼠发育的3个时间点 (E13胚胎13天 / P21出生21天 / P22出生22天)
的空间多组学数据。每个时间点包含：
  - 空间转录组 (RNA): 每个spot的基因表达
  - 染色质开放性 (ATAC): 每个spot的peak信号
  - 空间坐标 (x, y): spot在切片上的物理位置

【目标】
将3个时间点合并到一个统一的异构图中，用于训练GNN模型(类MarsGT)。
Gene/Peak 节点按名称跨时间点共享，Spot 节点带有时间标签。

【图结构 - 4种边关系】
  1. Spot-Spot:   KNN (k=5), 基于空间坐标 (仅各时间点内部, 不跨时间)
  2. Spot-Gene:   每个spot连接到top 50表达基因
  3. Spot-Peak:   每个spot连接到top 200开放peak
  4. Gene-Peak:   由 atac-{gene_name} 命名规则连接同名gene和peak

【预处理】
  - HVG (高可变基因): 每个时间点用scanpy(seurat方法)选5000个, 取并集
  - HVP (高可变peak): 每个时间点用scanpy(seurat方法)选10000个, 取并集

【最终图规模 (统一图)】
  节点: 13775 spots + 10192 genes + 16864 peaks = 40831
  边数: ~6.94M (Spot-Spot 40K, Spot-Gene 1.38M, Spot-Peak 5.51M, Gene-Peak 10.6K)
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')  # 屏蔽scanpy/pandas的版本警告，保持输出整洁

# ============================================================
# 1. 数据加载
# ============================================================

def load_dataset(data_dir):
    """
    加载单个时间点(E13/P21/P22)的表达矩阵、坐标和元数据。

    【输入文件格式】
    1) *_exprMatrix.tsv (表达矩阵)
       - 行=特征(gene或peak), 列=spot
       - 第一行可能是 [""] 或 ["gene"] + barcode列表 (取决于数据集)
       - 后续每行: feature_name + 各spot的表达/可达性值
       - 特征前缀区分两类:
           基因   : 常规基因名 (例: "Sox2", "Pax6")
           ATACpeak: "atac-" 前缀 (例: "atac-Sox2" 表示 Sox2 附近的peak)

    2) *_Spots.coords.tsv (坐标)
       - 无表头, 3列: barcode, x, y

    3) meta_*.tsv (元数据, 可选)
       - 含 RNA/ATAC/Joint 聚类标签 (4列或3列, 取决于时间点)

    【返回】
    dict 包含:
      rna_values     : (n_genes, n_spots) RNA表达矩阵
      atac_values    : (n_peaks, n_spots) ATAC可达性矩阵
      gene_names     : 基因名数组
      peak_names_raw : peak名数组 (含 'atac-' 前缀)
      spot_barcodes  : 表达矩阵中的spot barcode
      coord_barcodes : 坐标文件中的spot barcode
      coords         : (n_spots, 2) 空间坐标
      meta           : 元数据 DataFrame (可能为 None)
    """
    # 自动识别目录内的3类文件
    files = os.listdir(data_dir)
    tsv_files = [f for f in files if f.endswith('.tsv') and 'exprMatrix' in f]
    coord_files = [f for f in files if f.endswith('.tsv') and 'coords' in f]
    meta_files = [f for f in files if f.endswith('.tsv') and 'meta' in f]

    if not tsv_files or not coord_files:
        raise ValueError(f"在 {data_dir} 中找不到数据文件")

    expr_path = os.path.join(data_dir, tsv_files[0])
    coord_path = os.path.join(data_dir, coord_files[0])

    # 从目录名提取时间点标签 (例: data_E13 -> E13)
    prefix = os.path.basename(data_dir).replace('data_', '')
    print(f"  加载 {prefix} 数据集...")

    # --- 加载表达矩阵 ---
    # 注意: 不同时间点的header列名不同
    #   E13: 第一行 = [空字符串, barcode1, barcode2, ...]
    #   P21: 第一行 = ["gene", barcode1, barcode2, ...]
    # 用 header=0 让pandas自动把第一行作为列名
    with open(expr_path, 'r') as f:
        first_line = f.readline().strip()
    first_col_header = first_line.split('\t')[0]  # 记录但不直接使用

    df = pd.read_csv(expr_path, sep='\t', header=0)
    # 将第一列(特征名列)统一改名为 'feature', 与各时间点解耦
    df = df.rename(columns={df.columns[0]: 'feature'})

    feature_names = df['feature'].values       # 所有特征名 (gene + peak混合)
    spot_barcodes = df.columns[1:].values      # 列名 = spot barcode
    values = df.iloc[:, 1:].values             # 数值矩阵 (n_features, n_spots)
    values = values.astype(np.float32)         # 节省内存

    print(f"    特征数: {len(feature_names)}, Spot数: {len(spot_barcodes)}")

    # --- 拆分 RNA 和 ATAC ---
    # 凡是名字以 'atac-' 开头的都是 peak, 其余视为 gene
    atac_mask = np.array([name.startswith('atac-') for name in feature_names])
    gene_mask = ~atac_mask

    gene_names_raw = feature_names[gene_mask]    # 例: ["Sox2", "Pax6", ...]
    atac_names_raw = feature_names[atac_mask]    # 例: ["atac-Sox2", ...]
    rna_values = values[gene_mask]               # (n_genes, n_spots)
    atac_values = values[atac_mask]              # (n_peaks, n_spots)

    print(f"    基因: {len(gene_names_raw)}, ATAC peak: {len(atac_names_raw)}")

    # --- 加载坐标 ---
    # 坐标文件无表头，固定3列: barcode, x, y
    coord_df = pd.read_csv(coord_path, sep='\t', header=None)
    coord_df.columns = ['barcode', 'x', 'y']
    coord_barcodes = coord_df['barcode'].values

    # --- 加载元数据 (聚类标签等) ---
    # 不同时间点元数据列数不一致:
    #   E13/P21: 4列 (barcode, RNA_clusters, ATAC_clusters, Joint_clusters)
    #   P22:     3列 (无Joint_clusters)
    if meta_files:
        meta_path = os.path.join(data_dir, meta_files[0])
        meta_df = pd.read_csv(meta_path, sep='\t')
        # 若header缺失或不规范，则按列数重新指定列名
        if meta_df.columns[0].startswith('Unnamed') or 'barcode' not in meta_df.columns:
            meta_df = pd.read_csv(meta_path, sep='\t', header=None)
            n_cols = meta_df.shape[1]
            if n_cols == 4:
                meta_df.columns = ['barcode', 'RNA_clusters', 'ATAC_clusters', 'Joint_clusters']
            elif n_cols == 3:
                meta_df.columns = ['barcode', 'RNA_clusters', 'ATAC_clusters']
                meta_df['Joint_clusters'] = 'NA'  # 补全为缺失值
            meta_df = meta_df.iloc[1:]            # 跳过原始的"伪header"行
            meta_df = meta_df.reset_index(drop=True)
    else:
        meta_df = None

    return {
        'rna_values': rna_values,
        'atac_values': atac_values,
        'gene_names': gene_names_raw,
        'peak_names_raw': atac_names_raw,
        'spot_barcodes': spot_barcodes,
        'coord_barcodes': coord_barcodes,
        'coords': coord_df[['x', 'y']].values.astype(np.float32),
        'meta': meta_df,
    }


# ============================================================
# 2. 高可变特征筛选 (HVG / HVP)
# ============================================================

def select_hvg(rna_values, gene_names, n_top=5000):
    """
    用 scanpy 筛选高可变基因 (Highly Variable Genes, HVG)。

    【为什么需要HVG?】
    原始基因数(~16000-20000)中大部分是housekeeping基因或低表达噪声，
    保留方差/离散度高的基因可以:
      - 降低图规模, 减少GPU显存压力
      - 突出真正承载生物学信号的基因
      - 避免噪声基因干扰图的稀疏模式

    【方法】
    使用 seurat flavor: 基于均值-方差关系的离散度排序
    流程: 总数归一化(1e4) -> log1p -> 计算高可变性

    参数:
      rna_values : (n_genes, n_spots) 原始count矩阵
      gene_names : 基因名数组
      n_top      : 想保留的基因数 (实际可能略少)
    返回:
      selected_values : (n_selected, n_spots) 筛选后矩阵
      selected_genes  : 筛选后的基因名
      hvg_mask        : 布尔mask, 标记哪些原始基因被保留
    """
    import scanpy as sc
    import anndata as ad

    # scanpy 约定 AnnData 是 (cells, genes), 因此需要转置
    adata = ad.AnnData(rna_values.T)
    adata.var_names = gene_names

    # 标准预处理: 归一化 + log转换 (HVG选择基于log后的数据)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # seurat方法: 拟合均值-离散度关系, 取离散度最高的n_top个
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor='seurat')

    hvg_mask = adata.var['highly_variable'].values
    selected_genes = gene_names[hvg_mask]
    selected_values = rna_values[hvg_mask]  # 注意: 返回原始count, 不返回归一化后的

    print(f"    HVG: {n_top} -> {len(selected_genes)} 基因保留")
    return selected_values, selected_genes, hvg_mask


def select_hvp(atac_values, peak_names_raw, n_top=10000):
    """
    用 scanpy 筛选高可变 peak (Highly Variable Peaks, HVP)。

    与 HVG 流程一致, 仅参数(默认保留10000个)和数据不同。
    ATAC peak 数量(~24000)通常多于gene, 因此保留数也设得更大。

    参数:
      atac_values    : (n_peaks, n_spots) 原始count矩阵
      peak_names_raw : peak名数组 (含 atac- 前缀)
      n_top          : 想保留的peak数
    返回:
      selected_values, selected_peaks, hvp_mask (同上)
    """
    import scanpy as sc
    import anndata as ad

    adata = ad.AnnData(atac_values.T)
    adata.var_names = peak_names_raw

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor='seurat')

    hvp_mask = adata.var['highly_variable'].values
    selected_peaks = peak_names_raw[hvp_mask]
    selected_values = atac_values[hvp_mask]

    print(f"    HVP: {n_top} -> {len(selected_peaks)} peak 保留")
    return selected_values, selected_peaks, hvp_mask


# ============================================================
# 3. 建图 (4种边关系)
# ============================================================

def build_spot_spot_knn(coords, k=6, spot_start_id=0):
    """
    【边1/4】Spot-Spot: 基于空间坐标的 KNN 图。

    【动机】
    空间多组学中, 物理位置近的spot往往在生物学上也更相似 (同一组织结构),
    KNN边将这种空间先验注入图中, 让GNN能聚合邻近spot的信息。

    【实现】
    - 用欧氏距离找每个spot最近的 k+1 个spot (+1是因为最近的是自己)
    - 排除自环
    - 去重并对称化 (无向边)

    参数:
      coords        : (n_spots, 2) 空间坐标
      k             : 每个spot的邻居数
      spot_start_id : 全局节点ID偏移 (用于多时间点统一图中spot的全局编号)
    返回:
      edge_index : (2, E) numpy数组, 行0/行1为边的两端节点全局ID
    """
    n_spots = coords.shape[0]
    # min(k+1, n_spots) 防止spot数少于k+1时报错
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_spots), metric='euclidean')
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)  # indices: (n_spots, k+1)

    # 展开为边列表，剔除自环 (j != i)
    edges = []
    for i in range(n_spots):
        for j in indices[i]:
            if j != i:
                edges.append([spot_start_id + i, spot_start_id + j])

    edge_index = np.array(edges, dtype=np.int64).T  # (2, E)
    # np.sort + np.unique 实现:
    #   1) 把每条边的两端排序成 (small, large), 消除方向差异
    #   2) 沿列去重, 得到无向唯一边
    # 注: 这一步会让最终edge_index变成"上三角形式" (每条无向边只出现一次)
    edge_index = np.unique(np.sort(edge_index, axis=0), axis=1)

    print(f"    Spot-Spot KNN(k={k}): {edge_index.shape[1]} 条边")
    return edge_index


def build_spot_gene_edges(rna_values, top_k=50, spot_start_id=0, gene_start_id=None):
    """
    【边2/4】Spot-Gene: 每个spot连到其top_k表达的基因。

    【动机】
    每个spot是高维表达向量, 只保留top_k最强表达的基因相当于做局部稀疏化:
      - 保留spot最具特征的基因签名
      - 降低图密度, 避免每个spot连所有gene导致的过度连通

    【实现】
    - 对每个spot, 在表达>0的基因中取topK (按表达值排序)
    - 添加双向边 (spot↔gene), 因为是异构图但视作无向
    - 全局去重

    参数:
      rna_values    : (n_genes, n_spots) 表达矩阵
      top_k         : 每个spot最多选几个gene (默认50)
      spot_start_id : spot节点全局起始ID
      gene_start_id : gene节点全局起始ID (默认紧跟spot之后)
    返回:
      edge_index : (2, E), 每条边的两端节点全局ID
    """
    n_spots = rna_values.shape[1]
    n_genes = rna_values.shape[0]

    if gene_start_id is None:
        # 默认布局: spot 在前, gene 紧跟其后
        gene_start_id = spot_start_id + n_spots

    edges = []
    for s in range(n_spots):
        expr = rna_values[:, s]              # 该spot的所有基因表达 (n_genes,)
        nonzero = np.where(expr > 0)[0]      # 只考虑表达>0的基因
        if len(nonzero) == 0:
            continue                          # 该spot无表达, 跳过
        n_select = min(top_k, len(nonzero))  # 防止表达基因数<top_k
        # argsort返回升序索引, 加负号变降序, 取前n_select个
        top_idx = nonzero[np.argsort(-expr[nonzero])[:n_select]]
        for g in top_idx:
            # 添加双向边, GNN message passing时双向都需要
            edges.append([spot_start_id + s, gene_start_id + g])
            edges.append([gene_start_id + g, spot_start_id + s])

    edge_index = np.array(edges, dtype=np.int64).T
    edge_index = np.unique(edge_index, axis=1)  # 全局去重

    print(f"    Spot-Gene(top={top_k}): {edge_index.shape[1]} 条边")
    return edge_index


def build_spot_peak_edges(atac_values, peak_start_id, top_k=200, spot_start_id=0):
    """
    【边3/4】Spot-Peak: 每个spot连到其top_k开放的peak。

    【动机】
    同Spot-Gene, 选每个spot最强开放的peak表征其染色质状态。
    top_k=200 比gene的50更大, 因为ATAC信号更稀疏分散, 需要更多边覆盖。

    参数:
      atac_values   : (n_peaks, n_spots) ATAC可达性矩阵
      peak_start_id : peak节点全局起始ID (必填, 因为peak在spot+gene之后)
      top_k         : 每个spot最多选几个peak (默认200)
      spot_start_id : spot节点全局起始ID
    返回:
      edge_index : (2, E)
    """
    n_spots = atac_values.shape[1]
    n_peaks = atac_values.shape[0]

    edges = []
    for s in range(n_spots):
        acc = atac_values[:, s]              # 该spot所有peak的可达性
        nonzero = np.where(acc > 0)[0]
        if len(nonzero) == 0:
            continue
        n_select = min(top_k, len(nonzero))
        top_idx = nonzero[np.argsort(-acc[nonzero])[:n_select]]
        for p in top_idx:
            # 双向边
            edges.append([spot_start_id + s, peak_start_id + p])
            edges.append([peak_start_id + p, spot_start_id + s])

    edge_index = np.array(edges, dtype=np.int64).T
    edge_index = np.unique(edge_index, axis=1)

    print(f"    Spot-Peak(top={top_k}): {edge_index.shape[1]} 条边")
    return edge_index


def build_gene_peak_edges(gene_names, peak_names_raw, gene_start_id, peak_start_id):
    """
    【边4/4】Gene-Peak: 由命名规则 atac-{gene_name} ↔ {gene_name} 连接。

    【动机】
    数据集中peak的命名规则是 'atac-' + 邻近基因名 (上游调控区域),
    这种命名暗含 "peak调控该基因" 的先验知识 (cis-regulatory link)。
    通过这条边, GNN可以学习peak的开放性如何影响基因表达。

    【实现】
    - 剥掉 'atac-' 前缀得到peak对应的基因名
    - 查表: 该基因名是否在筛选后的gene_names中
    - 命中则连边 (双向)

    参数:
      gene_names     : 已筛选的基因名数组 (HVG并集)
      peak_names_raw : 已筛选的peak名数组 (含 atac- 前缀, HVP并集)
      gene_start_id  : gene节点全局起始ID
      peak_start_id  : peak节点全局起始ID
    返回:
      edge_index : (2, E)
    """
    # 步骤1: 从 'atac-{gene}' 剥离前缀, 得到每个peak对应的目标基因名
    peak_to_gene = np.array([name.replace('atac-', '', 1) for name in peak_names_raw])

    # 步骤2: 建立 gene名 -> gene_index 的查找表 (O(1)查询)
    gene_name_to_idx = {name: i for i, name in enumerate(gene_names)}

    edges = []
    matched = 0  # 统计有多少peak成功匹配到gene (HVG/HVP并集不一定完全对应)
    for p_idx, gene_name in enumerate(peak_to_gene):
        if gene_name in gene_name_to_idx:
            g_idx = gene_name_to_idx[gene_name]
            g_node = gene_start_id + g_idx     # gene的全局节点ID
            p_node = peak_start_id + p_idx     # peak的全局节点ID
            edges.append([g_node, p_node])
            edges.append([p_node, g_node])
            matched += 1

    edge_index = np.array(edges, dtype=np.int64).T
    edge_index = np.unique(edge_index, axis=1)

    print(f"    Gene-Peak: {matched}/{len(peak_names_raw)} 个peak匹配到gene, {edge_index.shape[1]} 条边")
    return edge_index


# ============================================================
# 4. 单时间点建图 (兼容老接口)
# ============================================================

def build_graph(data_dir, output_dir,
                k=6,
                top_genes=50,
                top_peaks=200,
                n_hvg=5000,
                n_hvp=10000):
    """
    单个时间点的完整建图流程 (用于独立分析单个数据集)。

    流程:
      1) load_dataset    : 读TSV
      2) select_hvg/hvp  : 筛选高可变特征
      3) build_*_edges   : 构建4种边
      4) 保存npz/npy到 output_dir

    参数说明见函数签名, 默认值与论文一致 (k=6 用于单图, k=5 用于多时间点合并)。
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: 加载数据 ---
    data = load_dataset(data_dir)

    # --- Step 2: HVG/HVP 筛选 ---
    print("  进行高可变特征筛选...")

    rna_selected, gene_names, gene_hvg_mask = select_hvg(data['rna_values'], data['gene_names'], n_top=n_hvg)
    atac_selected, peak_names, peak_hvp_mask = select_hvp(data['atac_values'], data['peak_names_raw'], n_top=n_hvp)

    n_spots = rna_selected.shape[1]
    n_genes = rna_selected.shape[0]
    n_peaks = atac_selected.shape[0]

    print(f"  最终节点数: Spot={n_spots}, Gene={n_genes}, Peak={n_peaks}")

    # --- 节点ID分配 (三段式布局) ---
    # 全局节点ID连续编号, 便于后续作为 PyG/DGL 的同质图处理:
    #   Spot:  [0, n_spots)
    #   Gene:  [n_spots, n_spots+n_genes)
    #   Peak:  [n_spots+n_genes, n_spots+n_genes+n_peaks)
    gene_start = n_spots
    peak_start = n_spots + n_genes

    # --- Step 3: 构建4种边 ---
    print("  构建边关系...")

    spot_spot_edges = build_spot_spot_knn(data['coords'], k=k)
    spot_gene_edges = build_spot_gene_edges(rna_selected, top_k=top_genes)
    spot_peak_edges = build_spot_peak_edges(atac_selected, peak_start, top_k=top_peaks)
    gene_peak_edges = build_gene_peak_edges(gene_names, peak_names,
                                            gene_start, peak_start)

    # --- Step 4: 保存 ---
    print(f"  保存到 {output_dir} ...")

    # 边和节点数 -> graph.npz (一个文件存所有图结构信息)
    np.savez(os.path.join(output_dir, 'graph.npz'),
             spot_spot_edges=spot_spot_edges,
             spot_gene_edges=spot_gene_edges,
             spot_peak_edges=spot_peak_edges,
             gene_peak_edges=gene_peak_edges,
             n_spots=n_spots,
             n_genes=n_genes,
             n_peaks=n_peaks)

    # 表达矩阵 -> 稀疏矩阵npz (节省空间, 稀疏度通常>90%)
    sp.save_npz(os.path.join(output_dir, 'rna_matrix.npz'),
                sp.csr_matrix(rna_selected))
    sp.save_npz(os.path.join(output_dir, 'atac_matrix.npz'),
                sp.csr_matrix(atac_selected))

    # 节点名称(用于后续可视化和回溯) + 空间坐标
    np.save(os.path.join(output_dir, 'spot_barcodes.npy'), data['spot_barcodes'])
    np.save(os.path.join(output_dir, 'gene_names.npy'), gene_names)
    np.save(os.path.join(output_dir, 'peak_names.npy'), peak_names)
    np.save(os.path.join(output_dir, 'coords.npy'), data['coords'])

    # 元数据(聚类标签)透传保存, 后续训练时可作监督信号
    if data['meta'] is not None:
        data['meta'].to_csv(os.path.join(output_dir, 'meta.csv'), index=False)

    # --- 统计信息 ---
    print(f"\n  建图完成! 节点统计:")
    print(f"    Spot: {n_spots}")
    print(f"    Gene: {n_genes}")
    print(f"    Peak: {n_peaks}")
    total_nodes = n_spots + n_genes + n_peaks
    total_edges = (spot_spot_edges.shape[1] +
                   spot_gene_edges.shape[1] +
                   spot_peak_edges.shape[1] +
                   gene_peak_edges.shape[1])
    print(f"    总节点: {total_nodes}, 总边数: {total_edges}")

    return {
        'n_spots': n_spots,
        'n_genes': n_genes,
        'n_peaks': n_peaks,
        'total_nodes': total_nodes,
        'total_edges': total_edges,
    }


# ============================================================
# 5. 多时间点统一建图 (核心流程)
# ============================================================

def build_combined_graph(data_dirs, output_dir,
                         k=5,
                         top_genes=50,
                         top_peaks=200,
                         n_hvg=5000,
                         n_hvp=10000):
    """
    将多个时间点 (E13/P21/P22) 合并到一张统一异构图中。

    【关键设计】
    - Gene/Peak 节点按名称跨时间点共享 (HVG/HVP 取并集)
      → 同一个基因在三个时间点都是同一个节点
    - Spot 节点不共享, 每个时间点的spot独立编号
      → 用 spot_time_labels 标记每个spot来自哪个时间点 (0/1/2)
    - Spot-Spot KNN 仅在同一时间点内部连接
      → 避免跨时间点的虚假空间邻接关系
    - Spot-Gene / Spot-Peak 自然跨时间点 (因为gene/peak共享)
      → GNN信息可经由共享的gene/peak节点在时间点间传递

    【7步流程】
      Step 1: 加载所有数据集
      Step 2: 每个时间点独立做 HVG/HVP
      Step 3: 取名字并集, 构建共享gene/peak节点集
      Step 4: 分配全局节点ID
      Step 5: 构建对齐到并集的合并表达矩阵 (稀疏)
      Step 6: 构建4种边
      Step 7: 保存

    参数:
      data_dirs : 多个时间点数据目录的列表
      output_dir: 输出目录
      k         : Spot-Spot KNN的邻居数 (合并图默认5)
      top_genes : Spot-Gene 每spot取多少基因
      top_peaks : Spot-Peak 每spot取多少peak
      n_hvg     : 每个时间点HVG数量
      n_hvp     : 每个时间点HVP数量
    返回:
      dict: 节点边数等统计信息
    """
    os.makedirs(output_dir, exist_ok=True)
    # 从目录名提取时间点标签 (data_E13 -> E13)
    time_names = [os.path.basename(d).replace('data_', '') for d in data_dirs]

    # ============================================================
    # Step 1: 加载所有数据集
    # ============================================================
    print("=" * 60)
    print("Step 1: 加载数据集")
    all_data = []
    for d in data_dirs:
        data = load_dataset(d)
        all_data.append(data)

    # ============================================================
    # Step 2: 每个时间点独立做 HVG/HVP
    # ============================================================
    # 为什么各自独立做? 因为每个时间点的细胞类型/表达分布不同,
    # 一起做HVG会让数量占优的时间点(P22)主导筛选结果
    print("\nStep 2: 高可变特征筛选 (每个时间点独立)")
    hv_genes_per_time = []        # list of (selected_values, selected_names)
    hv_peaks_per_time = []
    raw_gene_names_per_time = []  # 保留原始名字, 用于后续矩阵重组
    raw_peak_names_per_time = []

    for data in all_data:
        rna_sel, gene_names, _ = select_hvg(data['rna_values'], data['gene_names'], n_top=n_hvg)
        atac_sel, peak_names, _ = select_hvp(data['atac_values'], data['peak_names_raw'], n_top=n_hvp)
        hv_genes_per_time.append((rna_sel, gene_names))
        hv_peaks_per_time.append((atac_sel, peak_names))
        raw_gene_names_per_time.append(data['gene_names'])
        raw_peak_names_per_time.append(data['peak_names_raw'])

    # ============================================================
    # Step 3: 取名字并集, 构建共享节点集
    # ============================================================
    # 三时间点的HVG/HVP取并集, 任一时间点高可变即保留
    # → 在某些时间点低变的基因, 可能在其他时间点高变, 仍可被GNN利用
    print("\nStep 3: 构建共享节点集")
    all_gene_names = set()
    all_peak_names = set()
    for _, names in hv_genes_per_time:
        all_gene_names.update(names)
    for _, names in hv_peaks_per_time:
        all_peak_names.update(names)

    # sorted 保证不同运行得到相同的节点顺序 (可复现)
    gene_names_union = sorted(all_gene_names)
    peak_names_union = sorted(all_peak_names)
    # 反向查找表: 名字 -> 在并集中的索引
    gene_name_to_idx = {n: i for i, n in enumerate(gene_names_union)}
    peak_name_to_idx = {n: i for i, n in enumerate(peak_names_union)}

    G = len(gene_names_union)
    P = len(peak_names_union)
    print(f"  共享基因节点: {G} (并集)")
    print(f"  共享 Peak 节点: {P} (并集)")

    # ============================================================
    # Step 4: 节点ID布局
    # ============================================================
    # 全局ID分配:
    #   Spot:  [0, S_total)         按数据集顺序拼接 [E13的spot | P21的 | P22的]
    #   Gene:  [S_total, S_total+G) 字母序
    #   Peak:  [S_total+G, S_total+G+P) 字母序
    n_spots_per_time = [d['rna_values'].shape[1] for d in all_data]
    S_total = sum(n_spots_per_time)
    # cumsum + 前面补0: 得到每个时间点spot的全局起始ID
    # 例: [2187, 2373, 9215] -> [0, 2187, 4560]
    spot_start = np.cumsum([0] + n_spots_per_time[:-1])
    gene_start = S_total
    peak_start = S_total + G

    print(f"  Spot 总数: {S_total} (各时间点: {n_spots_per_time})")
    print(f"  节点ID: Spot 0..{S_total-1}, Gene {gene_start}..{gene_start+G-1}, "
          f"Peak {peak_start}..{peak_start+P-1}")

    # ============================================================
    # Step 5: 构建对齐到并集的合并表达矩阵
    # ============================================================
    # 难点: 不同时间点原始矩阵的行顺序 (基因排列) 不同, 需要重排对齐到 gene_names_union
    # 解法: 遍历每个时间点, 把每个原始基因的数据填到统一坐标系中
    print("\nStep 5: 构建合并表达矩阵")
    spot_time_labels = np.zeros(S_total, dtype=np.int32)  # 每个spot的时间标签 0/1/2

    # 用稀疏COO三元组构造大矩阵, 比直接申请dense高效
    rna_rows, rna_cols, rna_vals = [], [], []
    atac_rows, atac_cols, atac_vals = [], [], []

    for t, data in enumerate(all_data):
        s_start = spot_start[t]
        n_s = n_spots_per_time[t]
        # 标注: 这一段spot ID 都来自时间点t
        spot_time_labels[s_start:s_start + n_s] = t

        raw_genes = data['gene_names']
        raw_peaks = data['peak_names_raw']
        rna_raw = data['rna_values']    # (n_raw_genes, n_spots) 该时间点原始矩阵
        atac_raw = data['atac_values']  # (n_raw_peaks, n_spots)

        # 反查表: 原始名字 -> 在该时间点原始矩阵中的行号
        raw_gene_to_row = {n: i for i, n in enumerate(raw_genes)}
        raw_peak_to_row = {n: i for i, n in enumerate(raw_peaks)}

        # 对并集中的每个基因, 检查在该时间点是否存在, 存在则填值
        # 注意: 这里用双循环 (gene × spot), 大规模时较慢; 若性能瓶颈可矢量化重写
        for g_idx, g_name in enumerate(gene_names_union):
            if g_name in raw_gene_to_row:
                raw_row = raw_gene_to_row[g_name]
                for s in range(n_s):
                    val = rna_raw[raw_row, s]
                    if val > 0:  # 只存非零, 利用稀疏性
                        rna_rows.append(g_idx)
                        rna_cols.append(s_start + s)
                        rna_vals.append(val)

        # 同样处理ATAC peak
        for p_idx, p_name in enumerate(peak_names_union):
            if p_name in raw_peak_to_row:
                raw_row = raw_peak_to_row[p_name]
                for s in range(n_s):
                    val = atac_raw[raw_row, s]
                    if val > 0:
                        atac_rows.append(p_idx)
                        atac_cols.append(s_start + s)
                        atac_vals.append(val)

    # COO三元组 -> CSR稀疏矩阵 (适合行切片, 用于后续按时间点取子矩阵)
    rna_matrix = sp.csr_matrix((rna_vals, (rna_rows, rna_cols)),
                               shape=(G, S_total), dtype=np.float32)
    atac_matrix = sp.csr_matrix((atac_vals, (atac_rows, atac_cols)),
                                shape=(P, S_total), dtype=np.float32)
    print(f"  RNA 矩阵: {G} x {S_total}, {rna_matrix.nnz} 非零元素")
    print(f"  ATAC 矩阵: {P} x {S_total}, {atac_matrix.nnz} 非零元素")

    # ============================================================
    # Step 6: 构建4种边
    # ============================================================
    print("\nStep 6: 构建边关系")

    # ---- 6a. Spot-Spot KNN (每个时间点内部, 不跨时间) ----
    # 物理坐标是各切片独立的, 跨时间点没有可比性
    spot_spot_edges_list = []
    for t, data in enumerate(all_data):
        s_edges = build_spot_spot_knn(data['coords'], k=k,
                                       spot_start_id=spot_start[t])
        spot_spot_edges_list.append(s_edges)
    spot_spot_edges = np.concatenate(spot_spot_edges_list, axis=1)
    print(f"  Spot-Spot 总边数: {spot_spot_edges.shape[1]}")

    # ---- 6b. Spot-Gene (top 50, 每个时间点独立挑选) ----
    # 从合并矩阵中切出该时间点的spot列, 选top基因
    # 由于gene在矩阵中是统一的索引, 自然连接到共享的gene节点
    spot_gene_edges_list = []
    for t, data in enumerate(all_data):
        n_s = n_spots_per_time[t]
        # toarray() 把稀疏切片转成dense以便argsort
        rna_slice = rna_matrix[:, spot_start[t]:spot_start[t] + n_s].toarray()
        sg_edges = build_spot_gene_edges(rna_slice, top_k=top_genes,
                                          spot_start_id=spot_start[t],
                                          gene_start_id=gene_start)
        spot_gene_edges_list.append(sg_edges)
    spot_gene_edges = np.concatenate(spot_gene_edges_list, axis=1)
    print(f"  Spot-Gene 总边数: {spot_gene_edges.shape[1]}")

    # ---- 6c. Spot-Peak (top 200, 每个时间点独立挑选) ----
    spot_peak_edges_list = []
    for t, data in enumerate(all_data):
        n_s = n_spots_per_time[t]
        atac_slice = atac_matrix[:, spot_start[t]:spot_start[t] + n_s].toarray()
        sp_edges = build_spot_peak_edges(atac_slice, peak_start,
                                          top_k=top_peaks,
                                          spot_start_id=spot_start[t])
        spot_peak_edges_list.append(sp_edges)
    spot_peak_edges = np.concatenate(spot_peak_edges_list, axis=1)
    print(f"  Spot-Peak 总边数: {spot_peak_edges.shape[1]}")

    # ---- 6d. Gene-Peak (名字匹配, 全局共享, 只需建一次) ----
    # 因为gene和peak都是跨时间点共享的, 不需要按时间点重复
    gene_peak_edges = build_gene_peak_edges(gene_names_union, peak_names_union,
                                             gene_start, peak_start)

    # ============================================================
    # Step 7: 保存所有产物
    # ============================================================
    print(f"\nStep 7: 保存到 {output_dir}")

    # 主图结构文件: 4种边 + 节点统计 + 时间点信息
    np.savez(os.path.join(output_dir, 'graph.npz'),
             spot_spot_edges=spot_spot_edges,
             spot_gene_edges=spot_gene_edges,
             spot_peak_edges=spot_peak_edges,
             gene_peak_edges=gene_peak_edges,
             n_spots=S_total,
             n_genes=G,
             n_peaks=P,
             n_spots_per_time=np.array(n_spots_per_time),
             spot_start=np.array(spot_start),         # 后续可用于切分spot
             time_names=np.array(time_names))

    # 表达矩阵: 稀疏CSR格式
    sp.save_npz(os.path.join(output_dir, 'rna_matrix.npz'), rna_matrix)
    sp.save_npz(os.path.join(output_dir, 'atac_matrix.npz'), atac_matrix)

    # 节点元信息: 名称 + 坐标 + 时间标签
    all_barcodes = np.concatenate([d['spot_barcodes'] for d in all_data])
    all_coords = np.concatenate([d['coords'] for d in all_data])
    np.save(os.path.join(output_dir, 'spot_barcodes.npy'), all_barcodes)
    np.save(os.path.join(output_dir, 'gene_names.npy'), np.array(gene_names_union))
    np.save(os.path.join(output_dir, 'peak_names.npy'), np.array(peak_names_union))
    np.save(os.path.join(output_dir, 'coords.npy'), all_coords)
    np.save(os.path.join(output_dir, 'spot_time_labels.npy'), spot_time_labels)

    # 合并元数据(聚类标签等), 加上time_label列以便区分来源
    all_meta_list = []
    for t, d in enumerate(all_data):
        if d['meta'] is not None:
            m = d['meta'].copy()
            m['time_label'] = time_names[t]
            all_meta_list.append(m)
    if all_meta_list:
        combined_meta = pd.concat(all_meta_list, ignore_index=True)
        combined_meta.to_csv(os.path.join(output_dir, 'meta.csv'), index=False)

    # --- 统计输出 ---
    total_nodes = S_total + G + P
    total_edges = (spot_spot_edges.shape[1] + spot_gene_edges.shape[1] +
                   spot_peak_edges.shape[1] + gene_peak_edges.shape[1])
    print(f"\n  建图完成!")
    print(f"    Spot: {S_total} ({dict(zip(time_names, n_spots_per_time))})")
    print(f"    Gene: {G}, Peak: {P}")
    print(f"    总节点: {total_nodes}, 总边数: {total_edges:,}")

    return {
        'n_spots': S_total, 'n_genes': G, 'n_peaks': P,
        'total_nodes': total_nodes, 'total_edges': total_edges,
        'n_spots_per_time': n_spots_per_time,
    }


# ============================================================
# 6. 入口
# ============================================================

def main():
    """
    主入口: 对 E13/P21/P22 三个时间点构建统一图。

    输出目录: graph_combined/
    后续可由 visualize_graph.py 读取并生成可视化。
    """
    base_dir = '/Users/user/Desktop/任务/scg'
    data_dirs = [
        os.path.join(base_dir, 'data_E13'),
        os.path.join(base_dir, 'data_P21'),
        os.path.join(base_dir, 'data_P22'),
    ]
    output_dir = os.path.join(base_dir, 'graph_combined')

    # 调用合并建图. 参数与MarsGT论文设置一致:
    #   k=5: Spot-Spot KNN的近邻数
    #   top_genes=50, top_peaks=200: 每spot最多边数
    #   n_hvg=5000, n_hvp=10000: 各时间点保留特征数
    stats = build_combined_graph(data_dirs, output_dir,
                                  k=5,
                                  top_genes=50,
                                  top_peaks=200,
                                  n_hvg=5000,
                                  n_hvp=10000)

    print(f"\n{'='*60}")
    print("全部完成!")
    print(f"  Spots: {stats['n_spots']}, Genes: {stats['n_genes']}, "
          f"Peaks: {stats['n_peaks']}")
    print(f"  总节点: {stats['total_nodes']}, 总边数: {stats['total_edges']:,}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
