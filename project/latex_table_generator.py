import evaluate
import plotting
import pickle as pkl
from ir_measures import Success, MRR, nDCG, MAP, Precision, Recall

METRICS = [
    MRR(),
    nDCG @ 10,
    MAP(),
    Recall @ 10,
    Precision @ 10,
]

DATASETS = ["relish", "scidocs_cite"]

DATASET_META = {
    "relish": ("RELISH ranked recommendation dataset", "tab:metrics-relish"),
    "scidocs_cite": ("SciRepEval citation-triplets dataset", "tab:metrics-scirepeval"),
}


def top_2_scores(values):
    uniq = sorted(set(values), reverse=True)
    best = uniq[0]
    second = uniq[1] if len(uniq) > 1 else None
    return best, second


def val_to_cell(mean, lo, hi, best_val, second_val):
    mean_p, lo_p, hi_p = mean * 100.0, lo * 100.0, hi * 100.0
    txt = f"{mean_p:.1f} ({lo_p:.1f}--{hi_p:.1f})"
    if best_val is not None and mean == best_val:
        return f"\\textbf{{{txt}}}"
    if second_val is not None and mean == second_val:
        return f"\\underline{{{txt}}}"
    return txt


if __name__ == "__main__":
    with open(evaluate.RESULTS_DIR / "evaluation.pkl", "rb") as f:
        results = pkl.load(f)

    reranker_rows = ["None"] + evaluate.RERANKER_MODELS

    for dataset in DATASETS:
        best_second_by_metric = {}
        for metric in METRICS:
            vals = []
            for model_name in evaluate.EMBEDDING_MODELS:
                for reranker_name in reranker_rows:
                    entry = results[dataset][model_name][reranker_name][metric]
                    vals.append(float(entry["mean"]))
            best_second_by_metric[metric] = top_2_scores(vals)

        print("\\begin{table}[h]")
        print("  \\centering")
        print("  \\renewcommand{\\arraystretch}{1.25}")
        col_spec = "|l|l|" + "c|" * len(METRICS)
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print("\\hline")

        print("\\textbf{Model} & \\textbf{Reranker}", end="")
        for metric in METRICS:
            metric_name = plotting.METRIC_DISPLAY_NAMES[str(metric).split("@")[0]] + ("@" + str(metric).split("@")[1] if "@" in str(metric) else "")
            print(f" & \\textbf{{{metric_name}}}", end="")
        print("\\\\")
        print("\\hline")
        print("\\hline")

        for model_name in evaluate.EMBEDDING_MODELS:
            model_disp = plotting.DISPLAY_NAMES.get(model_name, model_name)

            for i, reranker_name in enumerate(reranker_rows):
                reranker_disp = plotting.DISPLAY_NAMES.get(reranker_name, reranker_name)

                if i == 0:
                    print(
                        f"\\multirow{{{len(reranker_rows)}}}{{*}}{{{model_disp}}} & {reranker_disp}",
                        end="",
                    )
                else:
                    print(f" & {reranker_disp}", end="")

                for metric in METRICS:
                    entry = results[dataset][model_name][reranker_name][metric]

                    mean = float(entry["mean"])
                    lo = float(entry.get("ci_lower", mean))
                    hi = float(entry.get("ci_upper", mean))
                    best_val, second_val = best_second_by_metric[metric]

                    print(
                        f" & {val_to_cell(mean, lo, hi, best_val, second_val)}", end=""
                    )

                print("\\\\")
                print("\\hline")
            if model_name != evaluate.EMBEDDING_MODELS[-1]:
                print("\\hline")

        print("\\end{tabular}")

        name, label = DATASET_META.get(dataset, (dataset, f"tab:{dataset}"))
        print(
            f"  \\caption{{Evaluation results on the {name} for all model combinations and selected metrics including the 90\\% confidence interval. Best results per metric are \\textbf{{bolded}}, second best are \\underline{{underlined}}.}}"
        )
        print(f"  \\label{{{label}}}")
        print("\\end{table}")
        print()
