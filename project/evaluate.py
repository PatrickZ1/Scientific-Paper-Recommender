from collections import Counter
from time import time
import faiss
import numpy as np
from ir_measures import RR, Qrel, ScoredDoc, Success, calc_aggregate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
from recomm_dataset import *
import pathlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

EMBEDDING_MODELS = [
    "sentence-transformers/stsb-roberta-base-v2",
    "pritamdeka/S-Scibert-snli-multinli-stsb",
    "sentence-transformers/allenai-specter",
]

# NOTE: the currently fine-tuned reranker model is only trained on a very small subset of the data for quick testing
#       For proper evaluation, please fine-tune on the full training set first!
RERANKER_MODELS = [
    "cross-encoder/ms-marco-TinyBERT-L2-v2",
    "./models/cross_encoder/tinybert_scidocs_cite",  # fine-tuned model
]

FAISS_CACHE_DIR = pathlib.Path("./.faiss_cache")

METRICS_PER_DATASET = {
    "relish": [
        Success(cutoff=1),
        Success(cutoff=5),
        Success(cutoff=10),
        Success(cutoff=20),
        RR(),
    ],
    "scidocs_cite": [
        Success(cutoff=1),
        Success(cutoff=5),
        Success(cutoff=10),
        Success(cutoff=20),
        RR(),
    ],
}


def to_safe_filename(model_name: str, dataset_name: str) -> str:
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(
        c if c in valid_chars else "_" for c in model_name + "_" + dataset_name
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

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(_eval_one_query, q) for q in queries]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Evaluating queries"
        ):
            run.extend(fut.result())

    agg = calc_aggregate(
        measures=METRICS_PER_DATASET[dataset_name], qrels=qrels, run=run
    )
    for measure in METRICS_PER_DATASET[dataset_name]:
        print(f"{measure}: {agg[measure]:.4f}")


if __name__ == "__main__":
    FAISS_CACHE_DIR.mkdir(exist_ok=True)

    # TODO: use more than 1/100 of the validation data for quick testing
    eval_sets = {
        "relish": relish_to_q_doc_qrel(load_relish()["evaluation"].shard(20, 0)),
        "scidocs_cite": scidoc_cite_to_q_doc_qrel(
            load_scidocs_cite()["validation"].shard(100, 0)
        ),
    }

    for eval_name, (queries, docs, qrels) in eval_sets.items():
        print(f"Evaluating on dataset: {eval_name}")
        print(f"Loaded {len(queries)} queries and {len(docs)} documents.")

        counts = np.array(list(Counter(qrel.query_id for qrel in qrels).values()))
        print("Statistics of documents per query:")
        print("Min:", counts.min())
        print("Max:", counts.max())
        print("Mean:", counts.mean())
        print("Median:", np.median(counts))

        for model_name in EMBEDDING_MODELS:
            print(f"Evaluating model: {model_name}")
            evaluate_model(eval_name, model_name, queries, docs, qrels)
            print("-" * 80)
            print()

            for reranker_name in RERANKER_MODELS:
                print(f"Evaluating model with reranker: {model_name} + {reranker_name}")
                evaluate_model(
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
