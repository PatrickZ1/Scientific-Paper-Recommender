from time import time
import faiss
from datasets import load_dataset
from ir_measures import RR, Qrel, ScoredDoc, Success, calc_aggregate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
from recomm_dataset import *

EMBEDDING_MODELS = [
    "sentence-transformers/allenai-specter",
    "sentence-transformers/stsb-roberta-base-v2",
    "pritamdeka/S-Scibert-snli-multinli-stsb",
]

RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"

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

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )

    embeddings.show_progress = True
    vector_store.add_documents(docs)  # approximately 320 docs per sec
    embeddings.show_progress = False

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
    queries, docs, qrels = scidoc_cite_to_q_doc_qrel(
        load_scidocs_cite()["validation"].shard(100, 0)
    )
    print(f"Loaded {len(queries)} queries and {len(docs)} documents.")

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
