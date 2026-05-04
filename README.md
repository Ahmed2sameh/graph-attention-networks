# Graph Attention Networks for Node Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmed2sameh/graph-attention-networks/blob/main/graph_attention_networks.ipynb)

> Implementing and comparing Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN) for node classification on the Cora citation dataset using PyTorch Geometric.

## 📌 Overview
This project implements two prominent Graph Neural Network architectures from scratch:
- **GAT** — Graph Attention Networks (Veličković et al., 2018)
- **GCN** — Graph Convolutional Networks (Kipf & Welling, 2017)

Both models are evaluated on the **Cora** citation network dataset for semi-supervised node classification.

## 🧰 Technologies Used
- **Python 3**
- **PyTorch** — Deep learning framework
- **PyTorch Geometric** — Graph neural network library
- **NetworkX + Matplotlib** — Graph visualization

## 📊 Dataset: Cora
| Property | Value |
|----------|-------|
| Nodes | 2,708 (papers) |
| Edges | 10,556 (citations) |
| Features | 1,433 (bag-of-words) |
| Classes | 7 (research topics) |
| Task | Semi-supervised node classification |

## 🏗️ Model Architectures

### Graph Attention Network (GAT)
```
GATConv(1433 → 8, heads=8) + ELU + Dropout(0.6)
    ↓
GATConv(64 → 7, heads=1, concat=False) + Dropout(0.6)
    ↓
Softmax — 7-class output
```

### Graph Convolutional Network (GCN)
```
GCNConv(1433 → 8) + ReLU + Dropout(0.5)
    ↓
GCNConv(8 → 7)
    ↓
Softmax — 7-class output
```

## 🚀 How to Run
1. Click **Open in Colab** above
2. Run all cells — PyTorch Geometric is installed automatically
3. GAT and GCN are trained and compared side by side

## 📂 Project Structure
```
├── graph_attention_networks.ipynb   # Full implementation: GAT + GCN + visualization
└── README.md
```

## 📚 References
- [Graph Attention Networks (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)
- [Semi-Supervised Classification with GCN (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907)

## 👤 Author
Ahmed Sameh — [GitHub](https://github.com/Ahmed2sameh)
