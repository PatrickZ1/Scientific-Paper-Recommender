import matplotlib.pyplot as plt
import numpy as np
import pathlib
import evaluate

DISPLAY_NAMES = {
    "cross-encoder/ms-marco-TinyBERT-L2-v2": "TinyBERT",
    "./models/cross_encoder/tinybert_scidocs_cite": "TinyBERT Fine Tuned",
    "sentence-transformers/allenai-specter": "Specter",
    "sentence-transformers/stsb-roberta-base-v2": "RoBERTa",
    "pritamdeka/S-Scibert-snli-multinli-stsb": "SciBERT",
    "relish": "Relish",
    "scidocs_cite": "SciDocs-Cite",
    "None": "None",
}

def plot_metric(dataset_name, dataset_results, metric_name, metric, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title(f"{metric_name} on {DISPLAY_NAMES[dataset_name]} Dataset")
    ax.set_xlabel(f"{metric_name}")
    
    x = np.arange(len(dataset_results) + 3)
    bar_width = 0.25

    for model_index, (model_name, model_results) in enumerate(dataset_results.items()):
        for reranker_index, (reranker_name, metrics) in enumerate(model_results.items()):
            value = metrics[metric]
            bar_position = x[model_index + 1] - (reranker_index - len(model_results)/2) * bar_width

            bar = ax.barh(bar_position, value, height=bar_width, label=DISPLAY_NAMES[reranker_name] if model_index == 0 else "", color=f"C{reranker_index}")
            ax.bar_label(bar, labels=[f"{value:.3f}"], label_type="edge", padding=3)

    ax.legend(title="Rerankers", loc='upper left')
    ax.set_yticks(x)
    ax.set_yticklabels([""] + [DISPLAY_NAMES[n] for n in dataset_results.keys()] + [""] * 2)           
    ax.grid(axis='x') 
    
    # add x range padding
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmax=xmax + 0.05 * (xmax - xmin))
    
    if save_path:
        plt.savefig(str(save_path / f"{dataset_name}_{metric_name}.png"))
    
    plt.show()


if __name__ == "__main__":
    save_path = pathlib.Path("figures")
    save_path.mkdir(exist_ok=True)
    
    eval_sets = {
        "relish": evaluate.relish_to_q_doc_qrel(evaluate.load_relish()["evaluation"].shard(20, 0)),
        "scidocs_cite": evaluate.scidoc_cite_to_q_doc_qrel(
            evaluate.load_scidocs_cite()["validation"].shard(100, 0)
        ),
    }
    
    eval_results = evaluate.evaluate_all(eval_sets)
    
    for dataset_name, dataset_results in eval_results.items():
         plot_metric(dataset_name, dataset_results, "Success@5", evaluate.Success@5, save_path=save_path)