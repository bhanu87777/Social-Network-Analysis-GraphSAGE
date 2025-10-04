import torch
from models.graphsage_model import GraphSAGE, train_full_batch
from utils.data_loader import load_cora
from config import GRAPH_SAGE_HIDDEN, GRAPH_SAGE_LR, GRAPH_SAGE_WEIGHT_DECAY, GRAPH_SAGE_EPOCHS, DEVICE

def layer_experiment(layer_list=[1,2,3,4]):
    dataset, data = load_cora()
    in_dim = dataset.num_node_features
    out_dim = dataset.num_classes
    results = {}
    for L in layer_list:
        print(f"\n=== Training GraphSAGE with {L} layers ===")
        model = GraphSAGE(in_channels=in_dim, hidden_channels=GRAPH_SAGE_HIDDEN, out_channels=out_dim, num_layers=L)
        opt = torch.optim.Adam(model.parameters(), lr=GRAPH_SAGE_LR, weight_decay=GRAPH_SAGE_WEIGHT_DECAY)
        best = train_full_batch(model, data, opt, epochs=GRAPH_SAGE_EPOCHS, device=DEVICE)
        results[L] = best
        print(f"Layers {L} -> Best test acc: {best:.4f}")
    for L, acc in results.items():
        print(f"Layers {L}: Test Acc = {acc:.4f}")
    return results