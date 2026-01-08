from time import time

import faiss
from datasets import load_dataset
from ir_measures import RR, Qrel, ScoredDoc, Success, calc_aggregate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODELS = [
    "sentence-transformers/allenai-specter",
    "sentence-transformers/stsb-roberta-base-v2",
    "pritamdeka/S-Scibert-snli-multinli-stsb",
]


def load_data(split="validation"):
    dataset = load_dataset("allenai/scirepeval", "cite_prediction_new", split=split)
    dataset = dataset.select(range(10000))  # TODO: remove limit for full eval
    queries, docs = {}, {}
    qrels = []

    for item in dataset:
        query = item["query"]
        query_id = str(query["corpus_id"])
        query_text = f"{query['title']} [SEP] {query['abstract']}"
        queries[query_id] = query_text

        pos = item["pos"]
        pos_id = str(pos["corpus_id"])
        pos_text = f"{pos['title']} [SEP] {pos['abstract']}"
        if pos_id not in docs:
            docs[pos_id] = pos_text

        neg = item["neg"]
        neg_id = str(neg["corpus_id"])
        neg_text = f"{neg['title']} [SEP] {neg['abstract']}"
        if neg_id not in docs:
            docs[neg_id] = neg_text

        qrels.append(Qrel(query_id=query_id, doc_id=pos_id, relevance=1))

    queries = [Document(page_content=text, id=qid) for qid, text in queries.items()]
    docs = [Document(page_content=text, id=did) for did, text in docs.items()]
    return queries, docs, qrels


def evaluate_model(model_name, queries, docs):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": "cuda"}
    )

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )
    vector_store.add_documents(docs)  # approximately 320 docs per sec

    run = []
    for query in queries:
        retrieved_docs = vector_store.similarity_search(query.page_content, k=20)
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
    queries, docs, qrels = load_data()
    print(f"Loaded {len(queries)} queries and {len(docs)} documents.")

    t1 = time()
    for model_name in EMBEDDING_MODELS:
        print(f"Evaluating model: {model_name}")
        evaluate_model(model_name, queries, docs)
    t2 = time()
    print(f"Total evaluation time: {t2 - t1:.2f} seconds")
