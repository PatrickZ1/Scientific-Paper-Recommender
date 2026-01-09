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

EMBEDDING_MODELS = [
    "pritamdeka/S-Scibert-snli-multinli-stsb",
    "sentence-transformers/allenai-specter",
    "sentence-transformers/stsb-roberta-base-v2",
]

RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"

FAISS_CACHE_DIR = pathlib.Path("./.faiss_cache")


def to_safe_filename(model_name: str) -> str:
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in valid_chars else "_" for c in model_name)


def evaluate_model(
    embedder_name: str,
    queries: list[Document],
    docs: list[Document],
    qrels: list[Qrel],
    k: int = 20,
    reranker: str | None = None,
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
    if reranker is not None:
        rerank_model = CrossEncoder(reranker, max_length=512, device="cuda")

    if (FAISS_CACHE_DIR / to_safe_filename(embedder_name)).exists():
        vector_store = FAISS.load_local(
            str(FAISS_CACHE_DIR / to_safe_filename(embedder_name)),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Loaded FAISS index from disk.")
        print(f"Number of documents in index: {vector_store.index.ntotal}")
        if vector_store.index.ntotal != len(docs):
            raise ValueError(
                "Number of documents in FAISS index does not match number of documents provided."
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

        vector_store.save_local(str(FAISS_CACHE_DIR / to_safe_filename(embedder_name)))
        print("Saved FAISS index to disk.")

    run = []
    for query in tqdm(queries, desc="Evaluating queries"):
        retrieved_docs = vector_store.similarity_search(query.page_content, k=k)

        if rerank_model is not None:
            # Rerank the retrieved documents
            doc_texts = [doc.page_content for doc in retrieved_docs]
            ranks = rerank_model.rank(query.page_content, doc_texts)
            retrieved_docs = [
                retrieved_docs[rank["corpus_id"]]
                for rank in sorted(ranks, key=lambda x: x["score"], reverse=True)
            ]

        for rank, doc in enumerate(retrieved_docs, start=1):
            run.append(ScoredDoc(query_id=query.id, doc_id=doc.id, score=-rank))

    measures = [
        Success(cutoff=1),
        Success(cutoff=5),
        Success(cutoff=10),
        Success(cutoff=20),
        RR(),
    ]
    agg = calc_aggregate(measures=measures, qrels=qrels, run=run)
    for measure in measures:
        print(f"{measure}: {agg[measure]:.4f}")


if __name__ == "__main__":
    FAISS_CACHE_DIR.mkdir(exist_ok=True)

    queries, docs, qrels = scidoc_cite_to_q_doc_qrel(
        load_scidocs_cite()["validation"].shard(100, 0)
    )
    print(f"Loaded {len(queries)} queries and {len(docs)} documents.")

    counts = np.array(list(Counter(qrel.query_id for qrel in qrels).values()))
    print("Statistics of documents per query:")
    print("Min:", counts.min())
    print("Max:", counts.max())
    print("Mean:", counts.mean())
    print("Median:", np.median(counts))

    t1 = time()
    for model_name in EMBEDDING_MODELS:
        print(f"Evaluating model: {model_name}")
        evaluate_model(model_name, queries, docs, qrels)
        print("-" * 80)
        print()
        print(f"Evaluating model with reranker: {model_name} + {RERANKER_MODEL}")
        evaluate_model(model_name, queries, docs, qrels, reranker=RERANKER_MODEL)
        print("-" * 80)
        print()
    t2 = time()
    print(f"Total evaluation time: {t2 - t1:.2f} seconds")
