import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# MarsGT-style figure: Heterogeneous graph + embedding visualization
# ============================================================

# Global style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 150,
})

TIME_COLORS = {'E13': '#5B9BD5', 'P21': '#70AD47', 'P22': '#FF6B6B'}

def load_graph_data(base_dir):
    """Load the combined graph."""
    graph_dir = os.path.join(base_dir, 'graph_combined')
    data = np.load(os.path.join(graph_dir, 'graph.npz'), allow_pickle=True)
    coords = np.load(os.path.join(graph_dir, 'coords.npy'))
    gene_names = np.load(os.path.join(graph_dir, 'gene_names.npy'), allow_pickle=True)
    peak_names = np.load(os.path.join(graph_dir, 'peak_names.npy'), allow_pickle=True)
    time_labels = np.load(os.path.join(graph_dir, 'spot_time_labels.npy'))
    time_names = data.get('time_names', np.array(['E13', 'P21', 'P22']))
    n_spots_per_time = data.get('n_spots_per_time', np.array([]))
    spot_start = data.get('spot_start', np.array([]))

    import pandas as pd
    meta = pd.read_csv(os.path.join(graph_dir, 'meta.csv'))

    return {
        'spot_spot': data['spot_spot_edges'],
        'spot_gene': data['spot_gene_edges'],
        'spot_peak': data['spot_peak_edges'],
        'gene_peak': data['gene_peak_edges'],
        'n_spots': int(data['n_spots']),
        'n_genes': int(data['n_genes']),
        'n_peaks': int(data['n_peaks']),
        'coords': coords,
        'gene_names': gene_names,
        'peak_names': peak_names,
        'time_labels': time_labels,
        'time_names': time_names,
        'n_spots_per_time': n_spots_per_time,
        'spot_start': spot_start,
        'meta': meta,
    }


# ============================================================
# Panel A: Heterogeneous graph network visualization
# ============================================================

def sample_region_subgraph(graph, time_idx=0, n_spots_sample=30):
    """Sample a small region from one time point for graph visualization."""
    spot_start = graph['spot_start'][time_idx]
    n_spots_t = graph['n_spots_per_time'][time_idx]
    coords = graph['coords'][spot_start:spot_start + n_spots_t]

    # Pick a central spot and its neighbors
    center = np.median(coords, axis=0)
    dists = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    central_idx = np.argmin(dists)

    # KNN-based expansion
    selected_spots = {central_idx}
    frontier = {central_idx}
    while len(selected_spots) < n_spots_sample:
        new_frontier = set()
        for s in frontier:
            for i in range(graph['spot_spot'].shape[1]):
                u = graph['spot_spot'][0, i] - spot_start
                v = graph['spot_spot'][1, i] - spot_start
                if u == s and v not in selected_spots:
                    selected_spots.add(v)
                    new_frontier.add(v)
                elif v == s and u not in selected_spots:
                    selected_spots.add(u)
                    new_frontier.add(u)
        if not new_frontier:
            break
        frontier = new_frontier

    selected_spots = list(selected_spots)[:n_spots_sample]
    selected_spots_global = [spot_start + s for s in selected_spots]

    # Find connected genes and peaks
    connected_genes = set()
    connected_peaks = set()
    gene_edges_list = []
    peak_edges_list = []
    ss_edges_list = []

    for s_global in selected_spots_global:
        for i in range(graph['spot_gene'].shape[1]):
            if graph['spot_gene'][0, i] == s_global:
                connected_genes.add(graph['spot_gene'][1, i])
                gene_edges_list.append((s_global, graph['spot_gene'][1, i]))
            elif graph['spot_gene'][1, i] == s_global:
                connected_genes.add(graph['spot_gene'][0, i])
                gene_edges_list.append((graph['spot_gene'][0, i], s_global))

        for i in range(graph['spot_peak'].shape[1]):
            if graph['spot_peak'][0, i] == s_global:
                connected_peaks.add(graph['spot_peak'][1, i])
                peak_edges_list.append((s_global, graph['spot_peak'][1, i]))
            elif graph['spot_peak'][1, i] == s_global:
                connected_peaks.add(graph['spot_peak'][0, i])
                peak_edges_list.append((graph['spot_peak'][0, i], s_global))

        for i in range(graph['spot_spot'].shape[1]):
            u = graph['spot_spot'][0, i]
            v = graph['spot_spot'][1, i]
            if u in selected_spots_global and v in selected_spots_global:
                ss_edges_list.append((u, v))

    # Limit and deduplicate
    connected_genes = list(connected_genes)[:8]
    connected_peaks = list(connected_peaks)[:8]
    gene_edges_list = [(s, g) for s, g in gene_edges_list if g in connected_genes][:30]
    peak_edges_list = [(s, p) for s, p in peak_edges_list if p in connected_peaks][:30]
    ss_edges_list = ss_edges_list[:60]

    # Add gene-peak edges
    gp_edges_list = []
    for i in range(graph['gene_peak'].shape[1]):
        u = graph['gene_peak'][0, i]
        v = graph['gene_peak'][1, i]
        if u in connected_genes and v in connected_peaks:
            gp_edges_list.append((u, v))
        elif v in connected_genes and u in connected_peaks:
            gp_edges_list.append((u, v))
    gp_edges_list = gp_edges_list[:10]

    return {
        'spots': selected_spots_global,
        'genes': connected_genes,
        'peaks': connected_peaks,
        'ss_edges': ss_edges_list,
        'sg_edges': gene_edges_list,
        'sp_edges': peak_edges_list,
        'gp_edges': gp_edges_list,
        'spot_start': spot_start,
        'time_idx': time_idx,
    }


def plot_heterogeneous_graph(ax, graph, sub):
    """Draw a MarsGT-style heterogeneous graph with 3 node types and 4 edge types."""
    spot_start = sub['spot_start']
    t_coords = graph['coords'][spot_start:spot_start + graph['n_spots_per_time'][sub['time_idx']]]

    # Assign layout positions
    pos = {}
    np.random.seed(42)

    # Spots: use their actual spatial coordinates, scaled and centered
    local_spots = [s - spot_start for s in sub['spots']]
    spot_coords = t_coords[local_spots]
    spot_x = spot_coords[:, 0]
    spot_y = spot_coords[:, 1]
    # Normalize
    spot_x = (spot_x - spot_x.mean()) * 0.35 / spot_x.std() if spot_x.std() > 0 else spot_x * 0
    spot_y = (spot_y - spot_y.mean()) * 0.35 / spot_y.std() if spot_y.std() > 0 else spot_y * 0

    for i, s_global in enumerate(sub['spots']):
        pos[s_global] = (spot_x[i], spot_y[i])

    # Genes: arrange in a ring on the left
    n_genes = len(sub['genes'])
    gene_angle = np.linspace(np.pi/2, 3*np.pi/2, n_genes, endpoint=True) if n_genes > 0 else []
    r_gene = 1.8
    for i, g in enumerate(sub['genes']):
        if i < len(gene_angle):
            pos[g] = (r_gene * np.cos(gene_angle[i]) - 1.5,
                      r_gene * np.sin(gene_angle[i]))

    # Peaks: arrange in a ring on the right
    n_peaks = len(sub['peaks'])
    peak_angle = np.linspace(-np.pi/2, np.pi/2, n_peaks, endpoint=True) if n_peaks > 0 else []
    r_peak = 1.8
    for i, p in enumerate(sub['peaks']):
        if i < len(peak_angle):
            pos[p] = (r_peak * np.cos(peak_angle[i]) + 1.5,
                      r_peak * np.sin(peak_angle[i]))

    # Draw edges first (behind nodes)
    for u, v in sub['ss_edges']:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#BDD7EE', linewidth=0.4, alpha=0.6, zorder=1)

    for u, v in sub['sg_edges']:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#A8D5BA', linewidth=0.4, alpha=0.5, zorder=1)

    for u, v in sub['sp_edges']:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#F4B4B4', linewidth=0.4, alpha=0.5, zorder=1)

    for u, v in sub['gp_edges']:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#E8D48B', linewidth=0.5, alpha=0.6, zorder=1,
                    linestyle='dashed')

    # Draw nodes
    spot_xs = []; spot_ys = []
    gene_xs = []; gene_ys = []
    peak_xs = []; peak_ys = []

    for node_id, (x, y) in pos.items():
        if node_id < graph['n_spots']:
            spot_xs.append(x); spot_ys.append(y)
        elif node_id < graph['n_spots'] + graph['n_genes']:
            gene_xs.append(x); gene_ys.append(y)
        else:
            peak_xs.append(x); peak_ys.append(y)

    if spot_xs:
        ax.scatter(spot_xs, spot_ys, s=20, c='#5B9BD5', edgecolors='#2E75B6',
                   linewidth=0.5, zorder=3, alpha=0.9)
    if gene_xs:
        ax.scatter(gene_xs, gene_ys, s=30, c='#70AD47', edgecolors='#548235',
                   linewidth=0.5, zorder=3, marker='s', alpha=0.9)
    if peak_xs:
        ax.scatter(peak_xs, peak_ys, s=30, c='#FF6B6B', edgecolors='#C0392B',
                   linewidth=0.5, zorder=3, marker='^', alpha=0.9)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#5B9BD5',
               markeredgecolor='#2E75B6', markersize=8, label='Spot'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#70AD47',
               markeredgecolor='#548235', markersize=8, label='Gene'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF6B6B',
               markeredgecolor='#C0392B', markersize=8, label='Peak'),
        Line2D([0], [0], color='#BDD7EE', linewidth=1.5, label='Spot-Spot (KNN)'),
        Line2D([0], [0], color='#A8D5BA', linewidth=1.5, label='Spot-Gene'),
        Line2D([0], [0], color='#F4B4B4', linewidth=1.5, label='Spot-Peak'),
        Line2D([0], [0], color='#E8D48B', linewidth=1.5, linestyle='dashed',
               label='Gene-Peak'),
    ]
    ax.legend(handles=legend_elements, fontsize=6.5, loc='lower left',
              ncol=2, framealpha=0.9, edgecolor='#CCCCCC')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')


# ============================================================
# Panel B: Spatial / cluster visualization of spots
# ============================================================

def get_cluster_colors(labels):
    unique = sorted(set(str(l) for l in labels))
    n = len(unique)
    if n <= 10:
        cmap = plt.cm.tab10
    elif n <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_rainbow
    return {u: cmap(i % cmap.N) for i, u in enumerate(unique)}


def plot_spatial_clusters(ax, graph, time_idx, show_legend=True):
    """Draw spatial spots colored by RNA cluster (MarsGT Panel B style)."""
    spot_start = graph['spot_start'][time_idx]
    n_s = graph['n_spots_per_time'][time_idx]
    t_name = graph['time_names'][time_idx]

    coords = graph['coords'][spot_start:spot_start + n_s]

    meta = graph['meta']
    # Determine which cluster column to use
    rna_col = 'RNA_clusters'
    joint_col = 'Joint_clusters' if 'Joint_clusters' in meta.columns else None

    if joint_col and meta[joint_col].iloc[0] != 'NA':
        labels_raw = meta[joint_col].iloc[spot_start:spot_start + n_s].values
        label_title = 'Joint clusters'
    else:
        labels_raw = meta[rna_col].iloc[spot_start:spot_start + n_s].values
        label_title = 'RNA clusters'

    import pandas as pd
    labels = np.array(['NA' if (isinstance(x, float) and pd.isna(x)) else str(x)
                       for x in labels_raw])
    color_map = get_cluster_colors(labels)

    for label_name in sorted(color_map.keys(), key=lambda x: (x != 'NA', x)):
        mask = labels == label_name
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[label_name]], s=2, alpha=0.75,
                   label=label_name, edgecolors='none')

    ax.set_title(f'{t_name} ({n_s} spots)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Spatial X'); ax.set_ylabel('Spatial Y')
    ax.set_aspect('equal')

    if show_legend:
        n_unique = len(color_map)
        if n_unique <= 25:
            ax.legend(fontsize=4.5, markerscale=2.5, loc='upper right',
                      bbox_to_anchor=(1.25, 1.0), title=label_title,
                      title_fontsize=6, ncol=max(1, n_unique // 15))


def plot_time_comparison_umap(ax, graph):
    """Generate a basic PCA/UMAP-like view by combining spatial coordinates across time."""
    # Since we don't have GNN embeddings yet, use spatial coordinates
    # with time-specific offsets to create a combined view
    coords_all = graph['coords']
    time_labels = graph['time_labels']
    time_names = graph['time_names']

    # Simple transformation: use spatial coords as-is, they naturally separate by time
    for t, t_name in enumerate(time_names):
        mask = time_labels == t
        ax.scatter(coords_all[mask, 0], coords_all[mask, 1],
                   c=TIME_COLORS.get(t_name, 'gray'), s=1, alpha=0.5,
                   label=f'{t_name} ({mask.sum()})')

    ax.set_title('Spatial embedding by time point', fontsize=10, fontweight='bold')
    ax.set_xlabel('Dim 1 (spatial X)'); ax.set_ylabel('Dim 2 (spatial Y)')
    ax.legend(fontsize=7, markerscale=4)


# ============================================================
# Main composite figure
# ============================================================

def create_marsgt_style_figure(base_dir, output_path):
    print("Loading graph data...")
    graph = load_graph_data(base_dir)
    time_names = graph['time_names']

    # Build figure: 3 rows × 3 cols layout
    # Row A: Heterogeneous graph (spans full width with subgraphs for each time point)
    # Row B: Spatial cluster visualization for E13, P21, P22
    # Row C: Combined overview

    fig = plt.figure(figsize=(16, 14))

    # ---- Title ----
    fig.suptitle('Heterogeneous Graph Construction for Spatial Multi-omics',
                 fontsize=14, fontweight='bold', y=0.98)

    # ---- Row A: Heterogeneous graph (3 time points) ----
    for t_idx, t_name in enumerate(time_names):
        ax = fig.add_subplot(3, 3, t_idx + 1)
        sub = sample_region_subgraph(graph, time_idx=t_idx, n_spots_sample=25)
        plot_heterogeneous_graph(ax, graph, sub)
        ax.set_title(f'Heterogeneous graph ({t_name})', fontsize=10,
                     fontweight='bold', loc='center')

    # ---- Row B: Spatial clusters per time point ----
    for t_idx, t_name in enumerate(time_names):
        ax = fig.add_subplot(3, 3, t_idx + 4)
        show_legend = (t_idx == 2)  # Only show legend for last panel
        plot_spatial_clusters(ax, graph, time_idx=t_idx, show_legend=show_legend)

    # ---- Row C: Combined views ----
    # C1: All spots by time
    ax_c1 = fig.add_subplot(3, 3, 7)
    plot_time_comparison_umap(ax_c1, graph)

    # C2: Node and edge summary
    ax_c2 = fig.add_subplot(3, 3, 8)
    n_spots = graph['n_spots']
    n_genes = graph['n_genes']
    n_peaks = graph['n_peaks']
    ss_e = graph['spot_spot'].shape[1]
    sg_e = graph['spot_gene'].shape[1]
    sp_e = graph['spot_peak'].shape[1]
    gp_e = graph['gene_peak'].shape[1]

    summary_text = (
        f"Graph Summary\n"
        f"{'─' * 30}\n"
        f"Nodes: {n_spots + n_genes + n_peaks:,}\n"
        f"  Spot: {n_spots:,}\n"
        f"  Gene: {n_genes:,}\n"
        f"  Peak: {n_peaks:,}\n\n"
        f"Edges: {ss_e + sg_e + sp_e + gp_e:,}\n"
        f"  Spot-Spot: {ss_e:,}\n"
        f"  Spot-Gene: {sg_e:,}\n"
        f"  Spot-Peak: {sp_e:,}\n"
        f"  Gene-Peak: {gp_e:,}\n\n"
        f"Spatial KNN: k = 5\n"
        f"Spot-Gene top: 50\n"
        f"Spot-Peak top: 200"
    )
    ax_c2.text(0.05, 0.95, summary_text, transform=ax_c2.transAxes,
               fontsize=8.5, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8,
                         edgecolor='#CCCCCC'))
    ax_c2.axis('off')

    # C3: Legend panel
    ax_c3 = fig.add_subplot(3, 3, 9)
    node_labels = [f'E13: {graph["n_spots_per_time"][0]:,} spots',
                   f'P21: {graph["n_spots_per_time"][1]:,} spots',
                   f'P22: {graph["n_spots_per_time"][2]:,} spots']
    colors_time = [TIME_COLORS[t] for t in time_names]

    y_pos = 0.9
    ax_c3.text(0.05, y_pos, 'Time Points', fontsize=9, fontweight='bold',
               transform=ax_c3.transAxes)
    for i, (label, c) in enumerate(zip(node_labels, colors_time)):
        y_pos -= 0.08
        ax_c3.add_patch(Circle((0.12, y_pos + 0.12), 0.025, color=c,
                                transform=ax_c3.transAxes))
        ax_c3.text(0.18, y_pos + 0.10, label, fontsize=8,
                   transform=ax_c3.transAxes, va='center')

    y_pos -= 0.06
    ax_c3.text(0.05, y_pos, 'Node Types', fontsize=9, fontweight='bold',
               transform=ax_c3.transAxes)
    node_info = [('● Spot', '#5B9BD5'), ('■ Gene', '#70AD47'), ('▲ Peak', '#FF6B6B')]
    for symbol, c in node_info:
        y_pos -= 0.08
        ax_c3.text(0.10, y_pos, symbol, fontsize=8, color=c,
                   transform=ax_c3.transAxes, va='center')

    y_pos -= 0.06
    ax_c3.text(0.05, y_pos, 'Edge Types', fontsize=9, fontweight='bold',
               transform=ax_c3.transAxes)
    edge_info = [
        ('Spot-Spot (KNN)', '#BDD7EE'),
        ('Spot-Gene', '#A8D5BA'),
        ('Spot-Peak', '#F4B4B4'),
        ('Gene-Peak (- -)', '#E8D48B'),
    ]
    for label, color in edge_info:
        y_pos -= 0.08
        ax_c3.add_line(Line2D([0.05, 0.20], [y_pos + 0.10, y_pos + 0.10],
                              color=color, linewidth=2.5,
                              transform=ax_c3.transAxes,
                              linestyle='dashed' if '(- -)' in label else 'solid'))
        ax_c3.text(0.22, y_pos + 0.08, label, fontsize=8,
                   transform=ax_c3.transAxes, va='center')

    ax_c3.set_xlim(0, 1); ax_c3.set_ylim(0, 1)
    ax_c3.axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def main():
    base = '/Users/user/Desktop/任务/scg'
    output_path = os.path.join(base, 'vis_combined', 'marsgt_style_figure.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_marsgt_style_figure(base, output_path)


if __name__ == '__main__':
    main()
