import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pathlib
import pickle as pkl
from ir_measures import Success, MRR, nDCG, MAP, Precision, Recall
import evaluate

RESULTS_FILE = pathlib.Path("./project/evaluation_results.pkl")

DISPLAY_NAMES = {
    "cross-encoder/ms-marco-TinyBERT-L2-v2": "TinyBERT",
    "./models/cross_encoder/tinybert_scidocs_cite": "TinyBERT Fine Tuned",
    "sentence-transformers/allenai-specter": "SPECTER2",
    "sentence-transformers/stsb-roberta-base-v2": "RoBERTa",
    "./models/embedding/roberta_scidocs_cite": "RoBERTa Fine Tuned",
    "pritamdeka/S-Scibert-snli-multinli-stsb": "SciBERT",
    "relish": "Relish",
    "scidocs_cite": "SciDocs-Cite",
    "None": "None",
}

METRIC_DISPLAY_NAMES = {
    "P": "Precision",
    "R": "Recall",
    "nDCG": "nDCG",
    "AP": "MAP",
    "RR": "MRR",
    "Success": "Success",
}

# Which metrics to plot per dataset
METRICS_PER_DATASET = {
    "relish": [
        # Success@1,
        Success @ 5,
        # Success@10,
        # Success@20,
        MRR(),
        nDCG @ 5,
        nDCG @ 10,
        MAP(),
        Precision @ 5,
        Precision @ 10,
        Recall @ 5,
        Recall @ 10,
    ],
    "scidocs_cite": [
        # Success@1,
        Success @ 5,
        # Success@10,
        # Success@20,
        MRR(),
        nDCG @ 5,
        nDCG @ 10,
        MAP(),
        Precision @ 5,
        Precision @ 10,
        Recall @ 5,
        Recall @ 10,
    ],
}


def plot_metric(dataset_name, dataset_results, metric, save_path=None):
    metric_name = METRIC_DISPLAY_NAMES[str(metric).split("@")[0]] + ("@" + str(metric).split("@")[1] if "@" in str(metric) else "")

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.set_title(
        f"{metric_name} on {DISPLAY_NAMES[dataset_name]} Dataset with 90% Confidence Intervals"
    )
    ax.set_xlabel(f"{metric_name}")

    x = np.arange(len(dataset_results) + 3)
    bar_width = 0.25

    for model_index, (model_name, model_results) in enumerate(dataset_results.items()):
        for reranker_index, (reranker_name, metric_values) in enumerate(model_results.items()):
            value = metric_values[metric]["mean"]
            lower = metric_values[metric]["ci_lower"]
            upper = metric_values[metric]["ci_upper"]
            bar_position = (
                x[model_index + 1]
                - (reranker_index - len(model_results) / 2) * bar_width
            )

            bar = ax.barh(
                bar_position,
                value,
                height=bar_width,
                xerr=[[value - lower], [upper - value]],
                error_kw={"capsize": 5},
                label=DISPLAY_NAMES[reranker_name] if model_index == 0 else "",
                color=f"C{reranker_index}",
            )
            ax.bar_label(bar, labels=[f"{value:.3f}"], label_type="edge", padding=3)

    ax.legend(title="Rerankers", loc="upper left")
    ax.set_yticks(x)
    ax.set_yticklabels(
        [""] + [DISPLAY_NAMES[n] for n in dataset_results.keys()] + [""] * 2
    )
    ax.grid(axis="x")

    # add x range padding
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmax=xmax + 0.05 * (xmax - xmin))

    if save_path:
        plt.savefig(str(save_path / f"{dataset_name}_{metric_name}.png"))
    else:
        plt.show()


def plot_metrics(dataset_name, dataset_results, metrics, save_path=None):
    metric_names = [METRIC_DISPLAY_NAMES[str(metric).split("@")[0]] + ("@" + str(metric).split("@")[1] if "@" in str(metric) else "") for metric in metrics]
    metric_name = metric_names[-1].split("@")[0]

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.set_title(f"{metric_name} on {DISPLAY_NAMES[dataset_name]} Dataset")
    ax.set_xlabel(f"{metric_name}")

    x = np.arange(len(dataset_results) + 3)
    bar_width = 0.25

    hatches = ["", "xx", "..", "//"]
    alphas = [1.0, 0.7, 0.5, 0.3]

    for model_index, (model_name, model_results) in enumerate(dataset_results.items()):
        for reranker_index, (reranker_name, metric_values) in enumerate(model_results.items()):
            values = [metric_values[metric]["full_ds"] for metric in metrics]

            bar_position = (
                x[model_index + 1]
                - (reranker_index - len(model_results) / 2) * bar_width
            )

            bar = None
            for metric_index, (value, metric_name) in enumerate(
                zip(values, metric_names)
            ):
                left = 0 if metric_index == 0 else values[metric_index - 1]
                plt.rcParams["hatch.color"] = f"C{reranker_index}"
                bar = ax.barh(
                    bar_position,
                    value - left,
                    height=bar_width,
                    label=(
                        DISPLAY_NAMES[reranker_name]
                        if model_index == 0 and metric_index == 0
                        else ""
                    ),
                    color=f"C{reranker_index}",
                    alpha=alphas[metric_index],
                    # edgecolor=f"C{reranker_index + 1}" if metric_index > 0 else None,
                    hatch=hatches[metric_index],
                    left=left,
                )
                plt.rcParams["hatch.color"] = "black"

            ax.bar_label(
                bar, labels=[f"{values[-1]:.3f}"], label_type="edge", padding=3
            )

    legend1 = ax.legend(title="Rerankers", loc="upper left")

    ax.legend(
        title="Metrics",
        loc="upper right",
        handles=[
            Patch(facecolor="white", hatch=hatch, edgecolor="black", label=label)
            for hatch, label in zip(hatches[: len(metrics)], metric_names)
        ],
    )
    ax.add_artist(legend1)

    ax.set_yticks(x)
    ax.set_yticklabels(
        [""] + [DISPLAY_NAMES[n] for n in dataset_results.keys()] + [""] * 2
    )
    ax.grid(axis="x")

    # add x range padding
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmax=xmax + 0.05 * (xmax - xmin))

    if save_path:
        plt.savefig(str(save_path / f"{dataset_name}_{'_'.join(metric_names)}.png"))
    else:
        plt.show()

def plot_evaluation_time(dataset_name, dataset_results, metric, save_path=None):
    metric_name = METRIC_DISPLAY_NAMES[str(metric).split("@")[0]] + ("@" + str(metric).split("@")[1] if "@" in str(metric) else "")

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.set_title(
        f"Evaluation Time on {DISPLAY_NAMES[dataset_name]}"
    )
    ax.set_ylabel(f"{metric_name}")
    ax.set_xlabel(f"Evaluation Time (seconds)")

    shapes = ["o", "^", "s", "D", "v", "P", "*", "X"]
    for model_index, (model_name, model_results) in enumerate(dataset_results.items()):
        for reranker_index, (reranker_name, metric_values) in enumerate(model_results.items()):
            name = f"{DISPLAY_NAMES[model_name]}"
            if reranker_name != "None":
                name += f" + {DISPLAY_NAMES[reranker_name]}"

            # print(metric_values['model_parameters'], model_name, reranker_name, metric_values['evaluation_time'], metric_values[metric]["mean"])
            
            scatter = ax.scatter(metric_values['evaluation_time'],
                                metric_values[metric]["mean"],
                                s=metric_values['model_parameters'] / 2_000_000,
                                label=name,
                                color=f"C{model_index}",
                                marker=shapes[reranker_index])

    ax.legend(title="Models", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(axis="both")
    
    fig.subplots_adjust(right=0.7)

    # xmin, xmax = ax.get_xlim()
    # ax.set_xlim(xmax=xmax + 0.05 * (xmax - xmin))

    if save_path:
        plt.savefig(str(save_path / f"{dataset_name}_evaluation_time_{metric_name}.png"))
    else:
        plt.show()


if __name__ == "__main__":
    save_path = pathlib.Path("figures")
    save_path.mkdir(exist_ok=True)

    # try:
    #     with open(RESULTS_FILE, "rb") as f:
    #         eval_results = pkl.load(f)
    # except FileNotFoundError:
    #     raise FileNotFoundError(
    #         f"Results file {RESULTS_FILE} not found. Please run evaluate.py first to generate evaluation results."
    #     )

    for dataset_name in ["relish", "scidocs_cite"]:
        dataset_results = {}
        
        for model_name in evaluate.EMBEDDING_MODELS:
            for reranker_name in ["None"] + evaluate.RERANKER_MODELS:
                try:
                    with open(evaluate.RESULTS_DIR / f"evaluation_{evaluate.to_safe_filename(dataset_name, model_name, reranker_name)}.pkl", "rb") as f:
                        results = pkl.load(f)
                        if model_name not in dataset_results:
                            dataset_results[model_name] = {}
                        dataset_results[model_name][reranker_name] = results
                except FileNotFoundError:
                    print(f"Skipping file for {dataset_name}, {model_name}, {reranker_name} (not found)")
                    continue
        
        for metric in METRICS_PER_DATASET[dataset_name]:
            plot_metric(dataset_name, dataset_results, metric, save_path=save_path)
            break
            # plot_evaluation_time(dataset_name, dataset_results, metric, save_path=save_path)

        # plot_metrics(
        #     dataset_name,
        #     dataset_results,
        #     [Success @ 1, Success @ 5, Success @ 10],
        #     save_path=save_path,
        # )

