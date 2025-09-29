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

def visualize_embedding(X, y, method='tsne', title='Feature Visualization', dim=2):
    """
    Reduces dimensionality of feature vectors and saves visualization into reports/figures.

    Args:
        X (ndarray): Feature matrix of shape [n_samples, n_features].
        y (array-like): Labels corresponding to each sample.
        method (str): 'tsne' or 'umap'.
        title (str): Title of the plot.
        dim (int): Number of dimensions to reduce to (2 or 3).

    Returns:
        Path to the saved image (Path).
    """
    if method == 'tsne':
        reducer = TSNE(n_components=dim, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=dim, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    X_embedded = reducer.fit_transform(X)

    # prepare output directory
    out_dir = Path('reports/figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = _safe_filename(f"{title}_{method}_{dim}_{timestamp}")
    save_path = out_dir / f"{base_name}.png"

    if dim == 2:
        df_vis = pd.DataFrame({
            'X1': X_embedded[:, 0],
            'X2': X_embedded[:, 1],
            'Label': y
        })
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_vis, x='X1', y='X2', hue='Label', palette='tab10', s=60)
        plt.title(title)
        plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
                   c=pd.factorize(y)[0], cmap='tab10', s=60)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    else:
        raise ValueError("dim must be 2 or 3")