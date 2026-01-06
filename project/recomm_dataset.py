from datasets import load_dataset

# Load these datasets: ~24Gb -> downloaded to cache dir: ~/.cache/huggingface/datasets/
# 1. SciDocs - Citation Prediction: Query + Positive + Negative samples
# 2. ReLiSH - Relevant Literature Search: Query + List of relevant documents with score
scidocs_cite = load_dataset("allenai/scirepeval", "cite_prediction_new")
relish = load_dataset("allenai/scirepeval", "relish")
