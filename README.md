<html lang="en">
<body>
  <header>
    <h1>Social Network Analysis using GraphSAGE</h1>
    <p><strong>Short description:</strong> A Social Network Analysis project using GraphSAGE on the Cora dataset for node classification and clustering. Compares GraphSAGE with DeepWalk and Logistic Regression, visualizing embeddings via PCA and evaluating model accuracy.</p>
    <hr/>
  </header>

  <section id="features">
    <h2>Features</h2>
    <ul>
      <li>GraphSAGE (full-batch) training for node classification on the Cora dataset.</li>
      <li>DeepWalk embedding generation + Logistic Regression baseline.</li>
      <li>Logistic Regression baseline directly on node features.</li>
      <li>PCA visualization of embeddings (saved as PNGs).</li>
      <li>Experiments:
        <ul>
          <li>Vary training-set size (show accuracy vs number of labeled samples).</li>
          <li>Vary number of GraphSAGE layers (observe oversmoothing effect).</li>
          <li>Compare fixed vs variable neighbor-sampling schedules (simple custom sampler).</li>
        </ul>
      </li>
      <li>Terminal outputs for accuracy and saved figures for cluster visualizations.</li>
    </ul>
  </section>

  <section id="file-structure">
    <h2>Repository structure</h2>
    <pre>
cora_graphsage_analysis/
│
├── data/
│   └── cora/ (auto-downloaded)
├── models/
│   ├── graphsage_model.py
│   ├── deepwalk_model.py
│   └── logistic_regression.py
├── utils/
│   ├── data_loader.py
│   ├── visualization.py
│   └── metrics.py
├── experiments/
│   ├── sampling_experiments.py
│   └── layer_experiments.py
├── config.py
├── main.py
├── requirements.txt
├── .gitignore
└── figures/  (generated PCA plots will be saved here)
    </pre>
  </section>

  <section id="requirements">
    <h2>Requirements</h2>
    <p>Install Python 3.8+ (3.10/3.11 recommended). Use a virtual environment. Key Python packages are listed in <code>requirements.txt</code>:</p>
    <pre>
torch
torch-geometric (and required torch-scatter, torch-sparse, torch-cluster, torch-spline-conv)
scikit-learn
matplotlib
numpy
pandas
networkx
gensim
tqdm
  </section>

  <section id="installation">
    <h2>Installation</h2>
    <ol>
      <li>Open a terminal and navigate to the project root:
        <pre>cd /path/to/cora_graphsage_analysis</pre>
      </li>
      <li>Create and activate a virtual environment (recommended):
        <pre>
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (cmd)

python -m venv .venv
.venv\Scripts\activate.bat

# macOS / Linux

python3 -m venv .venv
source .venv/bin/activate
</pre>
</li>
</ol>

  </section>

  <section id="run">
    <h2>How to run</h2>
    <p>Run the main script from the project root. Example:</p>
    <pre>python main.py</pre>
    <p>What this does:</p>
    <ul>
      <li>Downloads the Cora dataset to <code>data/cora/</code> (if not present).</li>
      <li>Runs baseline Logistic Regression on raw features, DeepWalk + Logistic Regression, and GraphSAGE training.</li>
      <li>Saves PCA plots for each method into <code>figures/</code> (e.g., <code>figures/GraphSAGE_embeddings.png</code>).</li>
      <li>Runs experiments for training-set size, number of GraphSAGE layers, and sampling-rate comparison; results are printed to the terminal.</li>
    </ul>
  </section>

  <section id="expected-output">
    <h2>Example terminal outputs & artifacts</h2>
    <p>Terminal prints will include progress and final accuracies, for example:</p>
    <pre>
Logistic Regression Test Accuracy: 0.7000
DeepWalk+Logistic Test Accuracy: 0.7400
GraphSAGE Test accuracy (best observed): 0.8200
Saved PCA plot to: figures/Logistic_on_raw_features.png
Saved PCA plot to: figures/DeepWalk_embeddings.png
Saved PCA plot to: figures/GraphSAGE_embeddings.png
Training proportion 0.05 (20 labels) -> Test Acc: 0.60
Training proportion 0.10 (40 labels) -> Test Acc: 0.68
...
Layers 1: Test Acc = 0.81
Layers 2: Test Acc = 0.82
Layers 3: Test Acc = 0.78
...
Sampling experiment results: {'fixed': ([10, 10], 0.80), 'variable': ([20, 5], 0.83)}
    </pre>
    <p>Generated figures are saved under <code>figures/</code>. Open them to inspect clustering by class (7 classes in Cora).</p>
  </section>

  <section id="conclusion">
    <h2>Conclusion</h2>
    <p>This repository demonstrates a small but complete workflow for node classification and clustering using GraphSAGE and comparative baselines (DeepWalk and logistic regression). It provides:</p>
    <ul>
      <li>end-to-end training and evaluation,</li>
      <li>visual inspection via PCA plots, and</li>
      <li>experiments to analyze how labeled sample size, network depth (layers), and sampling strategy affect accuracy.</li>
    </ul>
    <p>Use this as a starting point for further experiments: add other GNN architectures (GCN, GAT), larger datasets, or more sophisticated sampling (NeighborLoader or mini-batch training).</p>
  </section>
</body>
</html>
