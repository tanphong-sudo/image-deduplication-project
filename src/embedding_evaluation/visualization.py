import os
import re
from pathlib import Path
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import pandas as pd

def _safe_filename(s):
    s = re.sub(r'[^A-Za-z0-9_.-]', '_', s)
    return s

def visualize_embedding(X, y, title='Feature Visualization'):
    """
    Reduces dimensionality of feature vectors and saves visualization into reports/figures.

    Args:
        X (ndarray): Feature matrix of shape [n_samples, n_features].
        y (array-like): Labels corresponding to each sample.
        title (str): Title of the plot.

    Returns:
        Path to the saved image (Path).
    """
    
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Dimensionality reduction
    reducer = umap.UMAP(n_components=3, random_state=42)
    X_embedded = reducer.fit_transform(X)

    # Prepare output directory
    out_dir = Path('report/figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = _safe_filename(f"{title}_umap_{timestamp}")
    save_path = out_dir / f"{base_name}.png"

    # Plot 3D scatter
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
        c=pd.factorize(y)[0],
        cmap='tab10',
        s=60,
        alpha=0.6,            # giảm độ trong suốt để thấy điểm trùng
        edgecolors='k',       # thêm viền đen quanh mỗi điểm
        linewidths=0.5
    )
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path
