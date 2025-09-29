# config.py
SEED = 42
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# GraphSAGE hyperparams
GRAPH_SAGE_HIDDEN = 128
GRAPH_SAGE_EMBED_DIM = 128
GRAPH_SAGE_LR = 0.01
GRAPH_SAGE_WEIGHT_DECAY = 5e-4
GRAPH_SAGE_EPOCHS = 200

# DeepWalk hyperparams
DEEPWALK_DIM = 128
DEEPWALK_WALKS_PER_NODE = 10
DEEPWALK_WALK_LENGTH = 40
DEEPWALK_WINDOW = 5
DEEPWALK_EPOCHS = 5

# Logistic Regression
LR_MAX_ITER = 1000
