# GNNs for Alpha Signal Discovery in Stock Graphs

This project demonstrates the use of **Graph Neural Networks (GNNs)** to uncover structured "alpha" signals in a synthetic stock correlation graph. Nodes represent individual stocks, and edges simulate relationships (e.g., correlations). Labels are assigned using a rule-based structural condition, and a GCN is trained to infer these labels despite noisy input features.

The experiment shows how graph-based learning can outperform traditional models when the signal is embedded in relational patterns rather than individual node features.

![Prediction Visualization](assets/gcn_predictions.png)

## Features

- Synthetic stock graph generation using Erdős–Rényi model
- Rule-based alpha labeling (based on specific graph neighborhoods)
- Noisy 3D node features (returns, volatility, momentum)
- 2-layer GCN implementation using PyTorch Geometric
- Visualizations highlighting prediction correctness and stock labels

## Setup

1. Install dependencies in Google Colab or locally:
```bash
pip install torch torchvision torchaudio torch-geometric networkx matplotlib scikit-learn
```

2. Run `main.ipynb` in Google Colab to reproduce all results and visualizations.

## Medium Article

A full write-up of this project is available on Medium:

**[Graph Neural Networks (GNNs) for Alpha Signal Generation](https://medium.com/@soroushfarid/graph-neural-networks-gnns-for-alpha-signal-generation-28e056d76323)**

## Repository Structure

- `main.ipynb`: Full training, evaluation, and visualization code
- `assets/`: Contains the GCN prediction plot used in the README
- `README.md`: Project documentation

## Contributing

Feel free to fork the repo, modify the graph structure or feature generation, and explore applications to real-world market data or alternative labeling rules.
