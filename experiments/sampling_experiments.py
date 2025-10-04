import torch
import random
from models.graphsage_model import GraphSAGE
from utils.data_loader import load_cora, set_seeds
from config import GRAPH_SAGE_WEIGHT_DECAY, DEVICE

def build_adj_list(edge_index, num_nodes):
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[s].append(d)
    return adj

def sample_subgraph(adj, seed_nodes, sizes):
    nodes = set(seed_nodes)
    frontier = list(seed_nodes)
    all_nodes = set(nodes)
    for size in sizes:
        new_frontier = []
        for u in frontier:
            neighs = adj[u]
            if len(neighs) == 0:
                continue
            sampled = neighs if len(neighs) <= size else random.sample(neighs, size)
            for v in sampled:
                if v not in all_nodes:
                    new_frontier.append(v)
                    all_nodes.add(v)
        frontier = new_frontier
    nodes_sorted = sorted(list(all_nodes))
    node_id_map = {old: i for i, old in enumerate(nodes_sorted)}
    rows, cols = [], []
    for old_u in nodes_sorted:
        for old_v in adj[old_u]:
            if old_v in node_id_map:
                rows.append(node_id_map[old_u])
                cols.append(node_id_map[old_v])
    edge_index_sub = torch.tensor([rows, cols], dtype=torch.long) if len(rows) > 0 else torch.empty((2,0), dtype=torch.long)
    seed_pos = [node_id_map[s] for s in seed_nodes if s in node_id_map]
    seed_mask = torch.zeros(len(nodes_sorted), dtype=torch.bool)
    seed_mask[seed_pos] = True
    return nodes_sorted, node_id_map, edge_index_sub, seed_mask

def train_with_sampling(data, num_layers=2, sample_sizes=[10,10], epochs=100, hidden=128, lr=0.01):
    set_seeds()
    num_nodes = data.num_nodes
    adj = build_adj_list(data.edge_index, num_nodes)
    train_nodes = [int(i) for i in torch.where(data.train_mask.cpu())[0].tolist()]
    model = GraphSAGE(in_channels=data.num_features, hidden_channels=hidden, out_channels=int(data.y.max().item())+1, num_layers=num_layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=GRAPH_SAGE_WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        batch_seed = random.sample(train_nodes, k=min(64, len(train_nodes)))
        nodes_sorted, node_id_map, edge_index_sub, seed_mask = sample_subgraph(adj, batch_seed, sample_sizes)
        if len(nodes_sorted) == 0:
            continue
        x_sub = data.x[nodes_sorted].to(DEVICE)
        y_sub = data.y[nodes_sorted].to(DEVICE)
        edge_index_sub = edge_index_sub.to(DEVICE)
        opt.zero_grad()
        logits = model(x_sub, edge_index_sub)
        loss = loss_fn(logits[seed_mask], y_sub[seed_mask])
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits_full = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
                preds = logits_full.argmax(dim=1)
                test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Test Acc: {test_acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits_full = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
        preds = logits_full.argmax(dim=1)
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return model, test_acc

def sampling_rate_experiment(data, fixed_size=[10,10], variable_size=[20,5], num_layers=2):
    model_fix, acc_fix = train_with_sampling(data, num_layers=num_layers, sample_sizes=fixed_size, epochs=200)
    model_var, acc_var = train_with_sampling(data, num_layers=num_layers, sample_sizes=variable_size, epochs=200)
    print(f"Fixed sampling {fixed_size} -> Test Acc: {acc_fix:.4f}")
    print(f"Variable sampling {variable_size} -> Test Acc: {acc_var:.4f}")
    return {"fixed": (fixed_size, acc_fix), "variable": (variable_size, acc_var)}