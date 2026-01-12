import os
import pathlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
from ir_measures import Recall, Precision, MAP, nDCG, MRR, Qrel, ScoredDoc, Success, calc_aggregate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from recomm_dataset import *
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
import pickle as pkl
import time

EMBEDDING_MODELS = [
    "sentence-transformers/stsb-roberta-base-v2",
    "./models/embedding/roberta_scidocs_cite",  # fine-tuned model
    "pritamdeka/S-Scibert-snli-multinli-stsb",
    "sentence-transformers/allenai-specter",
]
RERANKER_MODELS = [
    "cross-encoder/ms-marco-TinyBERT-L2-v2",
    "./models/cross_encoder/tinybert_scidocs_cite",  # fine-tuned model
]

FAISS_CACHE_DIR = pathlib.Path("./.faiss_cache")

RESULTS_DIR = pathlib.Path("./project")

# How many bootstrap samples to use for confidence intervals
BOOTSTRAP_N = 1000

# Confidence interval bounds
LOWER_CI = 0.05
UPPER_CI = 0.95

# Which metrics to compute per dataset
METRICS_PER_DATASET = {
    "relish": [
        Success@1,
        Success@5,
        Success@10,
        Success@20,
        MRR(),
        nDCG@5,
        nDCG@10,
        MAP(),
        Precision@5,
        Precision@10,
        Recall@5,
        Recall@10,
    ],
    "scidocs_cite": [
        Success@1,
        Success@5,
        Success@10,
        Success@20,
        MRR(),
        nDCG@5,
        nDCG@10,
        MAP(),
        Precision@5,
        Precision@10,
        Recall@5,
        Recall@10,
    ],
}


def to_safe_filename(*names: list[str]) -> str:
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(
        c if c in valid_chars else "_" for c in "_".join(names)
    )


def evaluate_model(
    dataset_name: str,
    embedder_name: str,
    queries: list[Document],
    docs: list[Document],
    qrels: list[Qrel],
    k: int = 20,
    rerank_name: str = None,
):
    """
    Evaluate a given embedding model with FAISS and ir_measures.

    :param embedder_name: Name of the embedding model to use.
    :param queries: All queries for evaluation.
    :param docs: All documents for evaluation.
    :param qrels: Relevance judgments for evaluation.
    :param k: Number of top documents to retrieve.
    :param reranker: Optional reranker to refine the initial retrieval results.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=embedder_name, model_kwargs={"device": "cuda"}
    )

    rerank_model = None
    if rerank_name is not None:
        rerank_model = CrossEncoder(rerank_name, device="cuda")

    # If a cached FAISS index exists for this model and dataset, load it
    if (FAISS_CACHE_DIR / to_safe_filename(embedder_name, dataset_name)).exists():
        vector_store = FAISS.load_local(
            str(FAISS_CACHE_DIR / to_safe_filename(embedder_name, dataset_name)),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Loaded FAISS index from disk.")
        print(f"Number of documents in index: {vector_store.index.ntotal}")
        if vector_store.index.ntotal != len(docs):
            raise ValueError(
                "Number of documents in FAISS index does not match number of documents provided. Please delete the cached index and try again."
            )
        else:
            print("Number of documents matches the provided documents.")
    else:
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True,
        )

        embeddings.show_progress = True
        vector_store.add_documents(docs)
        embeddings.show_progress = False

        vector_store.save_local(
            str(FAISS_CACHE_DIR / to_safe_filename(embedder_name, dataset_name))
        )
        print("Saved FAISS index to disk.")

    run = []

    def _eval_one_query(query):
        """Evaluate a single query and return the scored documents."""
        retrieved_docs = vector_store.similarity_search(query.page_content, k=k)

        if rerank_model is not None:
            doc_texts = [doc.page_content for doc in retrieved_docs]
            ranks = rerank_model.rank(query.page_content, doc_texts)
            retrieved_docs = [
                retrieved_docs[r["corpus_id"]]
                for r in sorted(ranks, key=lambda x: x["score"], reverse=True)
            ]

        return [
            ScoredDoc(query_id=query.id, doc_id=doc.id, score=-rank)
            for rank, doc in enumerate(retrieved_docs, start=1)
        ]

    evaluation_start = time.time()

    # Evaluate the queries in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(_eval_one_query, q) for q in queries]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Evaluating queries"
        ):
            run.extend(fut.result())

    full_ds_eval = calc_aggregate(
        measures=METRICS_PER_DATASET[dataset_name], qrels=qrels, run=run
    )

    # Group qrels / run by query_id for bootstrap sampling
    qrels_by_qid = {}
    for qr in qrels:
        qrels_by_qid.setdefault(qr.query_id, []).append(qr)

    run_by_qid = {}
    for scored_doc in run:
        run_by_qid.setdefault(scored_doc.query_id, []).append(scored_doc)

    query_ids = list(qrels_by_qid.keys())
    np.random.seed(42)

    # Bootstrap sampling per query (not per document)
    # Ensures that all scored documents and qrels for a given query are included together
    # Calculate aggregate metrics for each bootstrap sample

    boot_aggregates = []
    for _ in tqdm(range(BOOTSTRAP_N), desc="Bootstrap sampling"):
        sampled_qids = np.random.choice(query_ids, size=len(query_ids), replace=True)

        boot_qrels = []
        boot_run = []

        # Rename query_ids to avoid collisions when the same query is sampled multiple times
        for i, qid in enumerate(sampled_qids):
            new_qid = f"{qid}_{i}"

            for qr in qrels_by_qid[qid]:
                boot_qrels.append(
                    Qrel(query_id=new_qid, doc_id=qr.doc_id, relevance=qr.relevance)
                )

            for scored_doc in run_by_qid.get(qid, []):
                boot_run.append(
                    ScoredDoc(
                        query_id=new_qid,
                        doc_id=scored_doc.doc_id,
                        score=scored_doc.score,
                    )
                )

        boot_aggregates.append(
            calc_aggregate(
                measures=METRICS_PER_DATASET[dataset_name],
                qrels=boot_qrels,
                run=boot_run,
            )
        )
    
    evaluation_time = time.time() - evaluation_start

    # Report mean and confidence intervals from the bootstrap samples
    summary = {}
    for measure in METRICS_PER_DATASET[dataset_name]:
        vals = np.array([float(ba[measure]) for ba in boot_aggregates], dtype=float)
        mean = float(vals.mean())
        lower = float(np.quantile(vals, LOWER_CI))
        upper = float(np.quantile(vals, UPPER_CI))
        summary[measure] = {
            "full_ds": float(full_ds_eval[measure]),
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
        }
        print(
            f"{measure}: full_ds={summary[measure]['full_ds']:.4f} "
            f"mean={mean:.4f} CI{round((UPPER_CI-LOWER_CI)*100)}%=[{lower:.4f}, {upper:.4f}]"
        )
    summary['evaluation_time'] = evaluation_time
    
    def parameter_count(model, trainable_only=False):
        return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)
    
    summary['model_parameters'] = parameter_count(SentenceTransformer(embedder_name)) + (parameter_count(rerank_model.model) if rerank_model is not None else 0)

    return summary

def evaluate_all(eval_sets):
    eval_results = {}

    for eval_name, (queries, docs, qrels) in eval_sets.items():
        print(f"Evaluating on dataset: {eval_name}")
        print("-" * 80)
        print(f"Loaded {len(queries)} queries and {len(docs)} documents.")

        counts = np.array(list(Counter(qrel.query_id for qrel in qrels).values()))
        print("Statistics of documents per query:")
        print("Min:", counts.min())
        print("Max:", counts.max())
        print("Mean:", counts.mean())
        print("Median:", np.median(counts))
        print()

        eval_results[eval_name] = {}

        for model_name in EMBEDDING_MODELS:
            eval_results[eval_name][model_name] = {}

            print("-" * 80)
            print(f"Evaluating model: {model_name}")
            print("-" * 80)
            eval_results[eval_name][model_name]['None'] = evaluate_model(eval_name, model_name, queries, docs, qrels)
            print("-" * 80)
            print()

            with open(RESULTS_DIR / f"evaluation_{to_safe_filename(eval_name, model_name, 'None')}.pkl", "wb") as f:
                pkl.dump(eval_results[eval_name][model_name]['None'], f)

            for reranker_name in RERANKER_MODELS:
                print("-" * 80)
                print(f"Evaluating model with reranker: {model_name} + {reranker_name}")
                print("-" * 80)

                eval_results[eval_name][model_name][reranker_name] = evaluate_model(
                    eval_name,
                    model_name,
                    queries,
                    docs,
                    qrels,
                    k=30,  # Use larger k when reranking
                    rerank_name=reranker_name,
                )
                print("-" * 80)
                print()    
                
                with open(RESULTS_DIR / f"evaluation_{to_safe_filename(eval_name, model_name, reranker_name)}.pkl", "wb") as f:
                    pkl.dump(eval_results[eval_name][model_name][reranker_name], f)

    return eval_results

if __name__ == "__main__":
    FAISS_CACHE_DIR.mkdir(exist_ok=True)

    eval_sets = {
        "relish": relish_to_q_doc_qrel(load_relish()["evaluation"]),
        "scidocs_cite": scidoc_cite_to_q_doc_qrel(
            load_scidocs_cite()["validation"]
        ),
    }

    results = evaluate_all(eval_sets)
    with open(RESULTS_DIR / f"evaluation.pkl", "wb") as f:
        pkl.dump(results, f)
