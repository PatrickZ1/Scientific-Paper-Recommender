import evaluate
import plotting
import pickle as pkl
from ir_measures import Success, MRR, nDCG, MAP, Precision, Recall

METRICS = {
    MRR(),
    nDCG @ 10,
    MAP(),
    Recall @ 5,
    Recall @ 10,
}

DATASETS = ["relish"]

if __name__ == "__main__":
    
    results = {}
    
    for dataset_name in DATASETS:
        results[dataset_name] = {}
        
        for model_name in evaluate.EMBEDDING_MODELS:
            for reranker_name in ['None'] + evaluate.RERANKER_MODELS:
                try:
                    with open(evaluate.RESULTS_DIR / f"evaluation_{evaluate.to_safe_filename(dataset_name, model_name, reranker_name)}.pkl", "rb") as f:
                        r = pkl.load(f)
                        if model_name not in results[dataset_name]:
                            results[dataset_name][model_name] = {}
                        results[dataset_name][model_name][reranker_name] = r
                except FileNotFoundError:
                    print(f"Skipping file for {dataset_name}, {model_name}, {reranker_name} (not found)")


    print("\\begin{tabular}{|l" + ("|" + "c" * len(METRICS)) * len(DATASETS) + "|}")
    print("\\hline")
    print("\multirow{2}{*}{Model}", end="")
    for dataset in DATASETS:
        print("& \multicolumn{" + str(len(METRICS)) + "}{|c|}{" + plotting.DISPLAY_NAMES[dataset] + "}", end="")
    print("\\\\")
    
    for dataset in DATASETS:
        for metric in METRICS:
            metric_name = plotting.METRIC_DISPLAY_NAMES[str(metric).split("@")[0]] + ("@" + str(metric).split("@")[1] if "@" in str(metric) else "")
            print(f" & {metric_name}", end="")
    print("\\\\")
    print("\\hline")
    
    for model_name in evaluate.EMBEDDING_MODELS:
        for reranker_name in ['None'] + evaluate.RERANKER_MODELS:
            
            display_name = plotting.DISPLAY_NAMES[model_name] if reranker_name == 'None' else f"{plotting.DISPLAY_NAMES[model_name]} + {plotting.DISPLAY_NAMES[reranker_name]}"
            print(display_name, end="")
            
            for dataset in DATASETS:
                for metric in METRICS:
                    if dataset not in results or model_name not in results[dataset] or reranker_name not in results[dataset][model_name]:
                        print(" & -", end="")
                        continue
                    
                    value = results[dataset][model_name][reranker_name][metric]["mean"]
                    print(f" & {value * 100:.1f}", end="")
            print("\\\\")
    print("\\hline")
    print("\\end{tabular}")
 