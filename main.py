import torch
import numpy as np
from utils.data_loader import load_cora, set_seeds 
from models.graphsage_model import GraphSAGE, train_full_batch 
from models.deepwalk_model import deepwalk_embedding 
from models.logistic_regression import train_eval_logistic  
from utils.visualization import pca_and_plot 
from config import *
from experiments.layer_experiments import layer_experiment
from experiments.sampling_experiments import sampling_rate_experiment


def run_all():
    set_seeds()
    dataset, data = load_cora()
    num_nodes = data.num_nodes
    in_dim = dataset.num_node_features
    out_dim = dataset.num_classes

    print("\n=== 1) Logistic Regression on raw features (baseline) ===")
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    acc_lr, report_lr = train_eval_logistic(X, y, train_mask, test_mask, max_iter=LR_MAX_ITER)
    print(f"Logistic Regression Test Accuracy: {acc_lr:.4f}")
    print(report_lr)

    print("\nPCA plot for raw features (Logistic Regression):")
    file_lr = pca_and_plot(X, y, title="Logistic_on_raw_features")
    print("Saved PCA plot to:", file_lr)

    print("\n=== 2) DeepWalk embeddings + Logistic Regression ===")
    emb = deepwalk_embedding(data.edge_index, num_nodes, dimensions=DEEPWALK_DIM, walks_per_node=DEEPWALK_WALKS_PER_NODE, walk_length=DEEPWALK_WALK_LENGTH, window=DEEPWALK_WINDOW, epochs=DEEPWALK_EPOCHS)
    acc_dw, report_dw = train_eval_logistic(emb, y, train_mask, test_mask, max_iter=LR_MAX_ITER)
    print(f"DeepWalk+Logistic Test Accuracy: {acc_dw:.4f}")
    print(report_dw)
    file_dw = pca_and_plot(emb, y, title="DeepWalk_embeddings")
    print("Saved PCA plot to:", file_dw)

    print("\n=== 3) GraphSAGE (full-batch) ===")
    model = GraphSAGE(in_channels=in_dim, hidden_channels=GRAPH_SAGE_HIDDEN, out_channels=out_dim, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=GRAPH_SAGE_LR, weight_decay=GRAPH_SAGE_WEIGHT_DECAY)
    best = train_full_batch(model, data, opt, epochs=GRAPH_SAGE_EPOCHS, device=DEVICE)
    print(f"GraphSAGE Test accuracy (best observed): {best:.4f}")

    # get embeddings for visualization
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data.x.to(DEVICE), data.edge_index.to(DEVICE)).cpu().numpy()
    file_gs = pca_and_plot(embeddings, y, title="GraphSAGE_embeddings")
    print("Saved PCA plot to:", file_gs)

    print("\n=== 4) Experiments: Training set size variation (accuracy vs number of labeled samples) ===")
    # We'll sample different proportions from the original training split and train GraphSAGE each time
    train_indices_all = torch.where(data.train_mask.cpu())[0].tolist()
    proportions = [0.05, 0.1, 0.2, 0.4, 1.0]
    results_size = {}
    for p in proportions:
        k = max(1, int(len(train_indices_all) * p))
        sampled = np.random.choice(train_indices_all, size=k, replace=False).tolist()
        # build a custom train_mask
        new_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        new_train_mask[sampled] = True
        # temporarily store original mask and swap
        orig_train_mask = data.train_mask.clone()
        data.train_mask = new_train_mask.to(data.train_mask.device)
        # train a new GraphSAGE (fresh weights)
        m = GraphSAGE(in_channels=in_dim, hidden_channels=GRAPH_SAGE_HIDDEN, out_channels=out_dim, num_layers=2).to(DEVICE)
        o = torch.optim.Adam(m.parameters(), lr=GRAPH_SAGE_LR, weight_decay=GRAPH_SAGE_WEIGHT_DECAY)
        best_acc = train_full_batch(m, data, o, epochs=100, device=DEVICE)
        results_size[p] = best_acc
        print(f"Training proportion {p:.2f} ({k} labels) -> Test Acc: {best_acc:.4f}")
        # restore original mask
        data.train_mask = orig_train_mask

    print("\nAccuracy vs training sample proportion:")
    for p, acc in results_size.items():
        print(f"{p:.2f}\t{acc:.4f}")

    print("\n=== 5) Experiments: Number of layers (oversmoothing) ===")
    _ = layer_experiment(layer_list=[1,2,3,4])

    print("\n=== 6) Sampling-rate experiment: fixed vs variable sample sizes ===")
    sampling_results = sampling_rate_experiment(data, fixed_size=[10,10], variable_size=[10,15], num_layers=2)
    print("Sampling experiment results:", sampling_results)

if __name__ == "__main__":
    run_all()
