import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def evaluate_embeddings_table(X, y):
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores and present as a table.

    Args:
        X (np.ndarray): Embedding vectors (n_samples x n_features)
        y (np.ndarray): Ground-truth labels

    Returns:
        pd.DataFrame: Table with metric names and values
    """
    data = []

    # Silhouette score (cao tốt)
    try:
        sil = silhouette_score(X, y)
        data.append(['Silhouette Score', sil, 'Higher is better'])
    except:
        data.append(['Silhouette Score', None, 'Higher is better'])

    # Davies-Bouldin index (thấp tốt)
    try:
        dbi = davies_bouldin_score(X, y)
        data.append(['Davies-Bouldin Index', dbi, 'Lower is better'])
    except:
        data.append(['Davies-Bouldin Index', None, 'Lower is better'])

    # Calinski-Harabasz index (cao tốt)
    try:
        chi = calinski_harabasz_score(X, y)
        data.append(['Calinski-Harabasz Index', chi, 'Higher is better'])
    except:
        data.append(['Calinski-Harabasz Index', None, 'Higher is better'])

    df = pd.DataFrame(data, columns=['Metric', 'Value', 'Interpretation'])
    return df

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 128)        # 100 ảnh, 128-dim embedding
    y = np.random.randint(0, 5, 100)   # 5 lớp

    df_scores = evaluate_embeddings_table(X, y)
    print(df_scores)
