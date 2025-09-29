# utils/visualization.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os

def pca_and_plot(embeddings, labels, title, outpath="figures", show_legend=True):
    """
    embeddings: (N, D)
    labels: (N,)
    Saves a PNG to outpath/<title>.png
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    pca = PCA(n_components=2)
    emb2 = pca.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    num_classes = np.max(labels) + 1
    for c in range(num_classes):
        idx = labels == c
        plt.scatter(emb2[idx,0], emb2[idx,1], label=str(c), alpha=0.7, s=10)
    plt.title(title)
    if show_legend:
        plt.legend(markerscale=2)
    plt.tight_layout()
    filename = os.path.join(outpath, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    return filename
