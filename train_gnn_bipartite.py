import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
import random
import argparse

# ---------- Config ----------
N_EPOCHS = 50
LR = 5e-4
EMB_DIM = 256           # final node embedding dim from GNN
HIDDEN_DIM = 512
BATCH_SIZE = 512        # number of positive edges per batch
NEG_PER_POS = 5         # number of negative replies per positive
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NODES_CSV = "./bipartite_graph/nodes.csv"
EDGE_INDEX_NPY = "./bipartite_graph/edge_index.npy"
# optional precomputed embeddings path next to nodes.csv (script will detect)
EMBEDDINGS_NPY = "./bipartite_graph/embeddings.npy"
OUT_DIR = "./gnn_output"
os.makedirs(OUT_DIR, exist_ok=True)
# --------------------------

# ---------- Utilities ----------
def load_graph():
    # load nodes
    nodes = pd.read_csv(NODES_CSV)
    # use dataframe index as node id unless explicit 'node_id' column exists
    if 'node_id' in nodes.columns:
        node_ids = nodes['node_id'].values
        # ensure index mapping exists
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    else:
        node_ids = nodes.index.values
        id_to_idx = {int(idx): int(idx) for idx in node_ids}

    node_type = nodes['node_type'].astype(int).values  # 0 = email, 1 = reply
    texts = nodes['text'].astype(str).values if 'text' in nodes.columns else None

    # load edges (expect shape [2, E] or [E,2])
    edge_arr = np.load(EDGE_INDEX_NPY)
    if edge_arr.shape[0] == 2:
        edge_index = edge_arr
    elif edge_arr.shape[1] == 2:
        edge_index = edge_arr.T
    else:
        raise ValueError("edge_index.npy shape not understood: " + str(edge_arr.shape))

    # embeddings: if present we will use them as initial node features; else compute using text
    if os.path.exists(EMBEDDINGS_NPY):
        print("Found embeddings.npy; loading as initial node features.")
        node_features = np.load(EMBEDDINGS_NPY)
        # check matching length
        assert node_features.shape[0] == len(node_type), "embeddings rows != nodes"
    else:
        # fallback: create simple random features (or compute sentence-transformer here)
        print("No embeddings.npy found. Creating random init features. (You can compute embeddings with sentence-transformers and save as embeddings.npy)")
        rng = np.random.RandomState(0)
        node_features = rng.normal(size=(len(node_type), EMB_DIM)).astype(np.float32)

    # build PyG Data
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index_t)
    # attach node_type and helpful masks
    data.node_type = torch.tensor(node_type, dtype=torch.long)
    return data, nodes

def train_val_test_edges(edge_index, node_type, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split positive edges into train/val/test. Only keep edges that go from email->reply.
    Returns dict with 'train','val','test' listing arrays of shape (E_split,2)
    """
    # edge_index: 2 x E
    src = edge_index[0]
    dst = edge_index[1]
    mask = (node_type[src] == 0) & (node_type[dst] == 1)  # email->reply only
    src_f = src[mask]
    dst_f = dst[mask]
    edges = np.vstack([src_f, dst_f]).T
    # shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(edges)
    n = len(edges)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    train_edges = edges[: n - n_val - n_test]
    val_edges = edges[n - n_val - n_test : n - n_test]
    test_edges = edges[n - n_test :]
    return {'train': train_edges, 'val': val_edges, 'test': test_edges}

# ---------- Model ----------
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, out_dim=EMB_DIM, dropout=0.25):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=8)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8)
        self.conv3 = GATConv(hidden_dim * 8, hidden_dim, heads=8)
        self.conv4 = GATConv(hidden_dim * 8, out_dim, heads=1)
        self.bn1 = BatchNorm1d(hidden_dim * 8)
        self.bn2 = BatchNorm1d(hidden_dim * 8)
        self.bn3 = BatchNorm1d(hidden_dim * 8)
        self.bn4 = BatchNorm1d(out_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = self.act(x2)
        x2 = self.dropout(x2)
        x2 = x2 + x1  # residual connection

        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = self.act(x3)
        x3 = self.dropout(x3)
        x3 = x3 + x2  # residual connection

        x4 = self.conv4(x3, edge_index)
        x4 = self.bn4(x4)
        # L2 normalize embedding for stable dot-product similarity
        x4 = F.normalize(x4, p=2, dim=-1)
        return x4

# ---------- Training utilities ----------
def generate_batch(edges, batch_size):
    # yield batches of positive edges
    n = len(edges)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    for i in range(0, n, batch_size):
        batch_idx = idxs[i:i+batch_size]
        yield edges[batch_idx]

def evaluate_ranking(embeddings, edges_eval, node_type, k_list=[1,5,10]):
    """
    embeddings: numpy [N, D]
    edges_eval: np.array (E,2) of positive edges (src, dst)
    For each src, compute scores to all reply nodes and rank
    Returns recall@k and MRR
    """
    N = embeddings.shape[0]
    node_type = node_type.numpy()
    reply_indices = np.where(node_type == 1)[0]
    emb_replies = embeddings[reply_indices]  # (R, D)
    recalls = {k: 0 for k in k_list}
    rr_total = 0.0
    for src, dst in edges_eval:
        e_src = embeddings[src]  # (D,)
        scores = emb_replies.dot(e_src)  # (R,)
        # rank descending
        order = np.argsort(-scores)
        # find rank of the true dst in reply_indices
        # find location where reply_indices[idx] == dst
        pos = np.where(reply_indices[order] == dst)[0]
        if len(pos) == 0:
            continue
        rank = pos[0] + 1
        rr_total += 1.0 / rank
        for k in k_list:
            if rank <= k:
                recalls[k] += 1
    mrr = rr_total / len(edges_eval)
    recall_scores = {k: recalls[k] / len(edges_eval) for k in k_list}
    return recall_scores, mrr

# ---------- Main training loop ----------
def main():
    print("Loading graph...")
    data, nodes_df = load_graph()
    data = data.to(DEVICE)
    node_type = data.node_type.to(DEVICE)
    print("Nodes:", data.num_nodes, "Edges:", data.num_edges)

    # train/val/test split on positive edges (email->reply)
    splits = train_val_test_edges(data.edge_index.cpu().numpy(), data.node_type.cpu(), val_ratio=0.1, test_ratio=0.1)
    train_edges = splits['train']
    val_edges = splits['val']
    test_edges = splits['test']
    print("Train/Val/Test edges:", len(train_edges), len(val_edges), len(test_edges))

    # model
    model = GraphSAGEEncoder(in_dim=data.x.shape[1], hidden_dim=HIDDEN_DIM, out_dim=EMB_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    best_val_mrr = 0.0
    for epoch in range(1, N_EPOCHS+1):
        model.train()
        epoch_loss = 0.0

        # train in positive-edge batches
        for batch in generate_batch(train_edges, BATCH_SIZE):
            # compute embeddings for full graph (full-batch) per batch to avoid backward graph reuse
            emb = model(data.x, data.edge_index)  # shape [N, D]

            # for each pos edge, sample negatives from reply set
            pos_src = torch.tensor(batch[:,0], dtype=torch.long, device=DEVICE)
            pos_dst = torch.tensor(batch[:,1], dtype=torch.long, device=DEVICE)

            # build candidates: for each pos we have 1 positive + NEG_PER_POS negatives
            # We'll score positives against a set of candidate replies and apply CE loss.
            # Candidate replies indices (R) shape: (batch_size, 1+NEG_PER_POS)
            reply_pool = torch.where(node_type == 1)[0].cpu().numpy()
            # sample negatives (avoid sampling the true pos dst)
            negs = []
            for true in pos_dst.cpu().numpy():
                candidates = [true]
                # sample negatives:
                while len(candidates) < 1 + NEG_PER_POS:
                    neg = int(np.random.choice(reply_pool))
                    if neg != true:
                        candidates.append(neg)
                negs.append(candidates)
            candidates = torch.tensor(negs, dtype=torch.long, device=DEVICE)  # (B, R)

            # compute scores: dot(emb[src], emb[candidates].T) for each row
            src_emb = emb[pos_src]  # (B, D)
            # gather candidate embeddings
            cand_emb = emb[candidates]  # (B, R, D)
            # compute dot product
            scores = (cand_emb * src_emb.unsqueeze(1)).sum(dim=-1)  # (B, R)
            # target is index 0 (positive)
            target = torch.zeros(scores.size(0), dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(scores, target)
            epoch_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # end epoch
        # evaluate
        model.eval()
        with torch.no_grad():
            emb_eval = model(data.x, data.edge_index).detach().cpu().numpy()
            recall_scores, mrr = evaluate_ranking(emb_eval, val_edges, data.node_type.cpu(), k_list=[1,5,10])

        print(f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | val MRR {mrr:.4f} | recall@1 {recall_scores[1]:.4f}")

        # save best
        if mrr > best_val_mrr:
            best_val_mrr = mrr
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_encoder.pt"))
            np.save(os.path.join(OUT_DIR, "best_embeddings.npy"), emb_eval)
            print("  -> saved best model (MRR improved)")

    # final eval on test
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_encoder.pt")))
    model.eval()
    with torch.no_grad():
        emb_test = model(data.x, data.edge_index).detach().cpu().numpy()
        recall_scores, mrr = evaluate_ranking(emb_test, test_edges, data.node_type.cpu(), k_list=[1,5,10])
    print("Final Test Results: MRR {:.4f} | R@1 {:.4f} | R@5 {:.4f} | R@10 {:.4f}".format(
        mrr, recall_scores[1], recall_scores[5], recall_scores[10]
    ))
    np.save(os.path.join(OUT_DIR, "final_embeddings.npy"), emb_test)

if __name__ == "__main__":
    main()
